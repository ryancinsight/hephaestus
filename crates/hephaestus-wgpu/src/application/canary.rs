//! Per-builtin capability canary for WGSL math builtins.
//!
//! naga lowers WGSL `exp`, `log`, `pow`, and `tanh` on the Vulkan backend to
//! the GLSL.std.450 extended instructions `Exp` (27), `Log` (28), `Pow` (26),
//! and `Tanh` (21) at every float width, but the GLSL.std.450 specification
//! admits only 16- or 32-bit floating-point operands for those instructions
//! (<https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html>).
//! An f64 operand is invalid SPIR-V, so every Vulkan driver's behavior on it
//! is undefined. The observed behaviors span the full range: on NVIDIA
//! RTX 5080 (driver 610.60) f64 `log`/`tanh` abort pipeline compilation, f64
//! `pow` access-violates at dispatch, and f64 `exp` computes at f32 grade; on
//! Mesa lavapipe (hosted CI) f64 `exp` reaches a worst relative error of 1. The
//! defect is the lowering, not an adapter, so the verdict is keyed by
//! backend and width rather than by adapter identity. Two mechanisms share
//! one query surface:
//!
//! - [`builtin_verdict`](crate::application::canary::builtin_verdict): the
//!   specification rule. f64 transcendentals on Vulkan report
//!   [`BuiltinVerdict::Blocked`](crate::application::canary::BuiltinVerdict)
//!   with the cause; every other pair, and a device without adapter info,
//!   reports supported.
//! - [`probe_precision`](crate::application::canary::probe_precision): a live
//!   one-dispatch measurement of a builtin's precision grade against CPU
//!   truth. It refuses blocked pairs before building any shader, so probing
//!   can never compile undefined SPIR-V or abort the process.
//!
//! Seams gate dispatch with [`require_expr_safe`](crate::application::canary::require_expr_safe)
//! (which scans the WGSL text for builtin calls, so composite expressions
//! like `GeluTanhOp` are covered without a manual op-to-builtin map) or the
//! explicit [`require_builtin`](crate::application::canary::require_builtin),
//! turning undefined results and process aborts into a typed
//! [`HephaestusError::Unsupported`](hephaestus_core::HephaestusError::Unsupported)
//! naming the cause.

use eunomia::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, Wgsl};
use std::any::TypeId;

use super::elementwise::encode_elementwise;
use super::pipeline::{try_cached_pipeline, workgroups};
use crate::infrastructure::device::WgpuDevice;

/// WGSL math builtins with a verdict rule and a live precision probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathBuiltin {
    /// `exp(x)`: GLSL.std.450 `Exp`, undefined at f64 on Vulkan.
    Exp,
    /// `log(x)`: GLSL.std.450 `Log`, undefined at f64 on Vulkan.
    Log,
    /// `pow(x, y)`: GLSL.std.450 `Pow`, undefined at f64 on Vulkan.
    Pow,
    /// `tanh(x)`: GLSL.std.450 `Tanh`, undefined at f64 on Vulkan.
    Tanh,
}

/// Element width under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CanaryWidth {
    /// 32-bit floats.
    F32,
    /// 64-bit floats.
    F64,
}

/// Map a Rust scalar to its canary width; every other type returns `None`.
///
/// This names the probe width of a bare scalar. The seam gates read the WGSL
/// type token instead, so vectors of f64 are refused as f64 is.
#[must_use]
pub fn width_of<T: 'static>() -> Option<CanaryWidth> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Some(CanaryWidth::F32)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        Some(CanaryWidth::F64)
    } else {
        None
    }
}

/// Verdict for one (device, builtin, width) triple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuiltinVerdict {
    /// No defect known; dispatch normally.
    Supported,
    /// Dispatches but computes below width precision. No current rule
    /// produces it: the one degraded pair once recorded (f64 `exp` on
    /// NVIDIA Vulkan) is undefined SPIR-V and now reports
    /// [`BuiltinVerdict::Blocked`].
    Degraded {
        /// Recorded worst-case relative error against CPU truth.
        max_relative_error: f64,
        /// What was measured, where, and when.
        cause: &'static str,
    },
    /// The lowering is undefined on this backend: compiling or dispatching
    /// may abort the process or return wrong values; refuse instead.
    Blocked {
        /// The violated contract and the behaviors observed.
        cause: &'static str,
    },
}

/// Cause recorded for one f64 builtin on Vulkan: the specification clause
/// plus every driver behavior observed for it.
const fn blocked_cause(builtin: MathBuiltin) -> &'static str {
    match builtin {
        MathBuiltin::Exp => {
            "naga lowers f64 `exp` to GLSL.std.450 Exp, which admits only 16/32-bit operands; observed f32-grade results on NVIDIA RTX 5080 (driver 610.60) and a worst relative error of 1 on Mesa lavapipe"
        }
        MathBuiltin::Log => {
            "naga lowers f64 `log` to GLSL.std.450 Log, which admits only 16/32-bit operands; observed pipeline-compile abort (0xc0000005) on NVIDIA RTX 5080 (driver 610.60)"
        }
        MathBuiltin::Pow => {
            "naga lowers f64 `pow` to GLSL.std.450 Pow, which admits only 16/32-bit operands; observed dispatch access violation (0xc0000005) on NVIDIA RTX 5080 (driver 610.60)"
        }
        MathBuiltin::Tanh => {
            "naga lowers f64 `tanh` to GLSL.std.450 Tanh, which admits only 16/32-bit operands; observed pipeline-compile abort (0x80000003) on NVIDIA RTX 5080 (driver 610.60)"
        }
    }
}

/// Whether `width` operands of the GLSL.std.450 precision-restricted
/// instructions are undefined on this device's backend.
///
/// The backend is read from the live `wgpu::Device`, so a device wrapped
/// without recorded adapter info is judged like an acquired one.
fn restricted_width_undefined(device: &WgpuDevice, width: CanaryWidth) -> bool {
    width == CanaryWidth::F64 && device.inner().adapter_info().backend == wgpu::Backend::Vulkan
}

/// Report the verdict for one builtin at one width on this device.
///
/// Every f64 builtin on the Vulkan backend is blocked: its SPIR-V lowering is
/// outside the GLSL.std.450 operand contract, so no driver result is defined.
/// Other backends and widths report [`BuiltinVerdict::Supported`].
#[must_use]
pub fn builtin_verdict(
    device: &WgpuDevice,
    builtin: MathBuiltin,
    width: CanaryWidth,
) -> BuiltinVerdict {
    if restricted_width_undefined(device, width) {
        BuiltinVerdict::Blocked {
            cause: blocked_cause(builtin),
        }
    } else {
        BuiltinVerdict::Supported
    }
}

/// Refuse dispatch when `builtin` is blocked at `width` on this device.
///
/// Degraded and supported pairs pass through: degradation is a precision
/// caveat for tolerance selection, not a dispatch refusal.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] naming the operation, builtin,
/// and cause when [`builtin_verdict`] blocks the pair.
pub fn require_builtin(
    device: &WgpuDevice,
    builtin: MathBuiltin,
    width: CanaryWidth,
    operation: &'static str,
) -> Result<()> {
    match builtin_verdict(device, builtin, width) {
        BuiltinVerdict::Blocked { cause } => Err(HephaestusError::Unsupported {
            message: format!(
                "{operation}: WGSL {builtin:?} at {width:?} is blocked on this backend: {cause}"
            ),
        }),
        BuiltinVerdict::Supported | BuiltinVerdict::Degraded { .. } => Ok(()),
    }
}

const TOKENS: &[(MathBuiltin, &str)] = &[
    (MathBuiltin::Exp, "exp("),
    (MathBuiltin::Log, "log("),
    (MathBuiltin::Pow, "pow("),
    (MathBuiltin::Tanh, "tanh("),
];

/// The remaining WGSL builtins naga lowers to GLSL.std.450 instructions
/// whose operands the specification restricts to 16 or 32 bits: `Radians`
/// (11) through `Atan2` (25) and `Exp2`/`Log2` (29, 30). They carry no live
/// probe, so they have no [`MathBuiltin`] variant; [`require_expr_safe`]
/// refuses them under the same rule as the probed four.
const RESTRICTED_TOKENS: &[&str] = &[
    "radians(", "degrees(", "sin(", "cos(", "tan(", "asin(", "acos(", "atan(", "sinh(", "cosh(",
    "asinh(", "acosh(", "atanh(", "atan2(", "exp2(", "log2(",
];

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Whether `source` calls `token` (a name followed by `(`) at an identifier
/// boundary, so `atanh(` never matches `tanh(`.
fn calls(source: &[u8], token: &str) -> bool {
    let token = token.as_bytes();
    source
        .windows(token.len())
        .enumerate()
        .any(|(start, window)| window == token && (start == 0 || !is_ident_byte(source[start - 1])))
}

/// Scan WGSL source for the canary builtins, honoring identifier boundaries.
///
/// `atanh(` must not match `tanh(`, and `log2(`/`log10(`/`lgamma(` must not
/// match `log(`; the template text is the ground truth of what the driver
/// will compile, so composite expressions are covered without a manual map.
#[must_use]
pub fn scan_builtins(source: &str) -> Vec<MathBuiltin> {
    TOKENS
        .iter()
        .filter(|(_, token)| calls(source.as_bytes(), token))
        .map(|(builtin, _)| *builtin)
        .collect()
}

/// Refuse dispatch when `source` uses a blocked builtin at `width`.
///
/// Callers pass the template or composed WGSL they are about to compile; each
/// [`MathBuiltin`] found by [`scan_builtins`] goes through [`require_builtin`],
/// and every other GLSL.std.450 precision-restricted builtin is refused under
/// the same backend-and-width rule.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] naming the operation and cause for
/// the first blocked builtin found.
pub fn require_expr_safe(
    device: &WgpuDevice,
    source: &str,
    width: CanaryWidth,
    operation: &'static str,
) -> Result<()> {
    for builtin in scan_builtins(source) {
        require_builtin(device, builtin, width, operation)?;
    }
    if restricted_width_undefined(device, width)
        && let Some(token) = RESTRICTED_TOKENS
            .iter()
            .find(|token| calls(source.as_bytes(), token))
    {
        let name = token.trim_end_matches('(');
        return Err(HephaestusError::Unsupported {
            message: format!(
                "{operation}: WGSL `{name}` at {width:?} is blocked on this backend: naga lowers it to a GLSL.std.450 instruction that admits only 16/32-bit operands"
            ),
        });
    }
    Ok(())
}

/// Canary width of the WGSL type a template is instantiated at, read from its
/// token so vector and composite types count by their component:
/// `vec2<f64>` is as undefined as `f64` under the GLSL.std.450 rule.
fn token_width(token: &str) -> Option<CanaryWidth> {
    if token.contains("f64") {
        Some(CanaryWidth::F64)
    } else if token.contains("f32") {
        Some(CanaryWidth::F32)
    } else {
        None
    }
}

/// Refuse dispatch when the WGSL template `expr`, instantiated at `T`, uses a
/// builtin blocked at `T`'s component width.
///
/// Types without an f32 or f64 component have no canary width and pass
/// through.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] as [`require_expr_safe`] does.
pub(crate) fn require_template_safe<T: DialectScalar<Wgsl>>(
    device: &WgpuDevice,
    expr: &str,
    operation: &'static str,
) -> Result<()> {
    token_width(T::TYPE_TOKEN).map_or(Ok(()), |width| {
        require_expr_safe(device, expr, width, operation)
    })
}

/// Measured precision grade of one live probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionGrade {
    /// Within a few ulps of CPU truth at width precision.
    Exact,
    /// f64 computing at f32 grade (worst error near f32 epsilon).
    SingleGrade,
    /// Worse than f32 grade: a broken lowering, driver, or probe.
    Poor,
}

/// Live measurement of one builtin against CPU truth.
#[derive(Clone, Copy, Debug)]
pub struct PrecisionReport {
    /// Builtin that was measured.
    pub builtin: MathBuiltin,
    /// Width that was measured.
    pub width: CanaryWidth,
    /// Probe samples dispatched.
    pub samples: u32,
    /// Worst relative error against CPU truth over all samples.
    pub max_relative_error: f64,
}

impl PrecisionReport {
    /// Map the measured error to its grade. Thresholds sit in the gaps:
    /// exact f64 lands near `1e-15`, f32-grade near `6e-8`, so the `1e-12` /
    /// `1e-5` cuts misclassify nothing. f32 truth is itself rounded, so the
    /// exact cut for f32 admits a few ulps of combined libm and rounding error.
    #[must_use]
    pub fn grade(&self) -> PrecisionGrade {
        let exact_cut = match self.width {
            CanaryWidth::F32 => 2e-7,
            CanaryWidth::F64 => 1e-12,
        };
        if self.max_relative_error <= exact_cut {
            PrecisionGrade::Exact
        } else if self.width == CanaryWidth::F64 && self.max_relative_error <= 1e-5 {
            PrecisionGrade::SingleGrade
        } else {
            PrecisionGrade::Poor
        }
    }
}

struct CanaryProbe;

fn probe_source<T: DialectScalar<Wgsl>>(builtin: MathBuiltin) -> String {
    let ty = T::TYPE_TOKEN;
    let (decls, body) = match builtin {
        MathBuiltin::Pow => (
            format!(
                "@group(0) @binding(0) var<storage, read> lhs: array<{ty}>;\n@group(0) @binding(1) var<storage, read> rhs: array<{ty}>;\n@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;"
            ),
            "pow(lhs[i], rhs[i])",
        ),
        MathBuiltin::Exp => (
            format!(
                "@group(0) @binding(0) var<storage, read> input: array<{ty}>;\n@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;"
            ),
            "exp(input[i])",
        ),
        MathBuiltin::Log => (
            format!(
                "@group(0) @binding(0) var<storage, read> input: array<{ty}>;\n@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;"
            ),
            "log(input[i])",
        ),
        MathBuiltin::Tanh => (
            format!(
                "@group(0) @binding(0) var<storage, read> input: array<{ty}>;\n@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;"
            ),
            "tanh(input[i])",
        ),
    };
    format!(
        "{decls}\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    let i = id.x;\n    if (i >= {n}u) {{ return; }}\n    out[i] = {body};\n}}",
        n = PROBE_LEN,
    )
}

const PROBE_LEN: usize = 8;

type TruthFn = fn(f64, f64) -> f64;
type ProbeInputs = ([f64; PROBE_LEN], [f64; PROBE_LEN], TruthFn);

/// Probe inputs plus the CPU-truth function, all in f64 space.
fn probe_inputs(builtin: MathBuiltin) -> ProbeInputs {
    match builtin {
        // Inputs stay in f32 range at both widths (e^100 would overflow f32).
        MathBuiltin::Exp => (
            [1.0, 0.5, -1.0, 2.0, 10.0, -10.0, 0.0, 20.0],
            [0.0; PROBE_LEN],
            (|x, _| x.exp()) as fn(f64, f64) -> f64,
        ),
        MathBuiltin::Log => (
            [
                1.0,
                core::f64::consts::E,
                0.5,
                2.0,
                10.0,
                100.0,
                0.1,
                1000.0,
            ],
            [0.0; PROBE_LEN],
            (|x, _| x.ln()) as fn(f64, f64) -> f64,
        ),
        MathBuiltin::Pow => (
            [2.0, 9.0, core::f64::consts::E, 4.0, 1.5, 10.0, 0.5, 3.0],
            [10.0, 0.5, 1.0, 3.0, 2.5, 2.0, 0.5, 3.0],
            f64::powf,
        ),
        MathBuiltin::Tanh => (
            [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 10.0],
            [0.0; PROBE_LEN],
            (|x, _| x.tanh()) as fn(f64, f64) -> f64,
        ),
    }
}

trait ScalarProbe: DialectScalar<Wgsl> + Pod + Copy {
    fn from_probe(value: f64) -> Self;
    fn rel_err(gpu: Self, cpu: f64) -> f64;
}

impl ScalarProbe for f32 {
    fn from_probe(value: f64) -> Self {
        value as f32
    }

    fn rel_err(gpu: Self, cpu: f64) -> f64 {
        ((f64::from(gpu) - cpu).abs() / cpu.abs().max(1e-30)).min(f64::INFINITY)
    }
}

impl ScalarProbe for f64 {
    fn from_probe(value: f64) -> Self {
        value
    }

    fn rel_err(gpu: Self, cpu: f64) -> f64 {
        ((gpu - cpu).abs() / cpu.abs().max(1e-300)).min(f64::INFINITY)
    }
}

fn probe_with<T: ScalarProbe>(
    device: &WgpuDevice,
    builtin: MathBuiltin,
    width: CanaryWidth,
) -> Result<PrecisionReport> {
    let (lhs_host, rhs_host, truth) = probe_inputs(builtin);
    let input = device.upload(&lhs_host.map(T::from_probe))?;
    let output = device.alloc_uninitialized::<T>(PROBE_LEN)?;
    let rhs_buffer: Option<_> = if builtin == MathBuiltin::Pow {
        Some(device.upload(&rhs_host.map(T::from_probe))?)
    } else {
        None
    };
    let mut entries = Vec::with_capacity(3);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: input.buffer.as_entire_binding(),
    });
    if let Some(rhs) = &rhs_buffer {
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: rhs.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: output.buffer.as_entire_binding(),
        });
    } else {
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: output.buffer.as_entire_binding(),
        });
    }
    let key = (
        TypeId::of::<CanaryProbe>(),
        TypeId::of::<T>(),
        builtin as u32,
    );
    let pipeline = try_cached_pipeline(device, key, "hephaestus-canary", || {
        probe_source::<T>(builtin)
    })?;
    let groups = workgroups(PROBE_LEN, BlockWidth::DEFAULT)?;
    encode_elementwise(device, &pipeline, "hephaestus-canary", &entries, groups)?;
    let actual = device.download_owned(&output)?;
    let mut max_relative_error = 0.0_f64;
    for (index, gpu) in actual.iter().enumerate() {
        let expected = truth(lhs_host[index], rhs_host[index]);
        max_relative_error = max_relative_error.max(T::rel_err(*gpu, expected));
    }
    Ok(PrecisionReport {
        builtin,
        width,
        samples: PROBE_LEN as u32,
        max_relative_error,
    })
}

/// Measure one builtin's precision grade live against CPU truth.
///
/// Blocked pairs are refused by [`builtin_verdict`] before any shader is
/// built, so probing never compiles undefined SPIR-V or aborts the process. f64 probing needs a
/// device with the `ShaderF64` feature; without it pipeline creation fails
/// with a typed validation error.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] for blocked pairs, and
/// surfaces pipeline, dispatch, or transfer failures otherwise.
pub fn probe_precision(
    device: &WgpuDevice,
    builtin: MathBuiltin,
    width: CanaryWidth,
) -> Result<PrecisionReport> {
    if let BuiltinVerdict::Blocked { cause } = builtin_verdict(device, builtin, width) {
        return Err(HephaestusError::Unsupported {
            message: format!("canary refuses to compile {builtin:?} at {width:?}: {cause}"),
        });
    }
    match width {
        CanaryWidth::F32 => probe_with::<f32>(device, builtin, width),
        CanaryWidth::F64 => probe_with::<f64>(device, builtin, width),
    }
}

#[cfg(test)]
#[path = "tests_canary.rs"]
mod tests;
