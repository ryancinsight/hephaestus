//! Per-builtin driver capability canary for WGSL.
//!
//! Spec-valid WGSL compiles and runs exactly on most drivers, but three f64
//! transcendental builtins are driver-defective on the recorded NVIDIA Vulkan
//! adapter (RTX 5080, vendor `0x10DE`, device `0x2C02`): `log` and `tanh`
//! abort pipeline compilation (`0xc0000005` / `0x80000003`, bare-wgpu repro
//! with zero repo code on the stack), and `pow` passes validation then
//! access-violates at dispatch. A fourth, `exp`, dispatches but computes at
//! f32 grade (`e^1` equals `f32(e)` bit-for-bit). The templates are spec-valid,
//! so a blanket static gate would wrongly disable good drivers; the full fix
//! is an out-of-process dynamic probe (a crash-class compile cannot be probed
//! in-process without aborting the parent). Until then this module pairs two
//! mechanisms behind one query surface:
//!
//! - [`builtin_verdict`]: a recorded table keyed by exact adapter identity
//!   (backend, vendor, device). Known-bad triples report [`BuiltinVerdict`];
//!   every unknown adapter (or device without adapter info) default-allows,
//!   preserving current behavior where nothing is proven.
//! - [`probe_precision`]: a live one-dispatch measurement of a builtin's
//!   precision grade against CPU truth. It refuses crash-class pairs up front
//!   (via the table) so probing itself can never abort the process.
//!
//! Seams gate crash-class dispatch with [`require_expr_safe`] (which scans the
//! op template text for builtin calls, so composite expressions like
//! `GeluTanhOp` are covered without a manual op-to-builtin map) or the
//! explicit [`require_builtin`], turning process aborts into typed
//! [`HephaestusError::Unsupported`] naming the cause.

use eunomia::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, Wgsl};
use std::any::TypeId;

use super::elementwise::encode_elementwise;
use super::pipeline::{try_cached_pipeline, workgroups};
use crate::infrastructure::device::WgpuDevice;

/// WGSL math builtins with recorded or probeable driver behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathBuiltin {
    /// `exp(x)`: degraded to f32 grade at f64 on the recorded adapter.
    Exp,
    /// `log(x)`: aborts pipeline compilation at f64 on the recorded adapter.
    Log,
    /// `pow(x, y)`: passes validation, access-violates at dispatch at f64.
    Pow,
    /// `tanh(x)`: aborts pipeline compilation at f64 on the recorded adapter.
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

/// Map a Rust scalar to its canary width; non-float scalars return `None`.
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

/// Verdict for one (adapter, builtin, width) triple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuiltinVerdict {
    /// No defect recorded; dispatch normally.
    Supported,
    /// Dispatches but computes below width precision.
    Degraded {
        /// Recorded worst-case relative error against CPU truth.
        max_relative_error: f64,
        /// What was measured, where, and when.
        cause: &'static str,
    },
    /// Compiling or dispatching aborts/crashes the process; refuse instead.
    Blocked {
        /// What was observed, where, and when.
        cause: &'static str,
    },
}

struct Finding {
    backend: wgpu::Backend,
    vendor: u32,
    device: u32,
    builtin: MathBuiltin,
    width: CanaryWidth,
    verdict: BuiltinVerdict,
}

/// Crash- and degradation-class pairs proven by probe. Keyed by exact adapter
/// identity; entries carry their provenance and are removed only by deliberate
/// re-proof (a driver update may fix the defect, but assuming so risks a
/// process abort, so the safe direction is to keep the block).
const FINDINGS: &[Finding] = &[
    Finding {
        backend: wgpu::Backend::Vulkan,
        vendor: 0x10DE,
        device: 0x2C02,
        builtin: MathBuiltin::Log,
        width: CanaryWidth::F64,
        verdict: BuiltinVerdict::Blocked {
            cause: "NVIDIA Vulkan on RTX 5080 (driver 610.60) aborts compiling naga-valid f64 `log` (0xc0000005; bare-wgpu repro, keystone 2026-10-08)",
        },
    },
    Finding {
        backend: wgpu::Backend::Vulkan,
        vendor: 0x10DE,
        device: 0x2C02,
        builtin: MathBuiltin::Tanh,
        width: CanaryWidth::F64,
        verdict: BuiltinVerdict::Blocked {
            cause: "NVIDIA Vulkan on RTX 5080 (driver 610.60) aborts compiling naga-valid f64 `tanh` (0x80000003; bare-wgpu repro, keystone 2026-10-08)",
        },
    },
    Finding {
        backend: wgpu::Backend::Vulkan,
        vendor: 0x10DE,
        device: 0x2C02,
        builtin: MathBuiltin::Pow,
        width: CanaryWidth::F64,
        verdict: BuiltinVerdict::Blocked {
            cause: "NVIDIA Vulkan on RTX 5080 (driver 610.60) passes f64 `pow` validation then access-violates at dispatch (0xc0000005; scalar-power 2026-10-07)",
        },
    },
    Finding {
        backend: wgpu::Backend::Vulkan,
        vendor: 0x10DE,
        device: 0x2C02,
        builtin: MathBuiltin::Exp,
        width: CanaryWidth::F64,
        verdict: BuiltinVerdict::Degraded {
            max_relative_error: 6e-8,
            cause: "NVIDIA Vulkan on RTX 5080 (driver 610.60) computes f64 `exp` at f32 grade: e^1 equals f32(e) bit-for-bit (attention/fusion 2026-10-07)",
        },
    },
];

/// Report the recorded verdict for one builtin at one width on this device.
///
/// Unknown adapters and devices without adapter info default to
/// [`BuiltinVerdict::Supported`]: the table only ever blocks proven-bad
/// triples, never speculative ones.
#[must_use]
pub fn builtin_verdict(
    device: &WgpuDevice,
    builtin: MathBuiltin,
    width: CanaryWidth,
) -> BuiltinVerdict {
    let Some(info) = device.adapter_info() else {
        return BuiltinVerdict::Supported;
    };
    FINDINGS
        .iter()
        .find(|entry| {
            entry.backend == info.backend
                && entry.vendor == info.vendor
                && entry.device == info.device
                && entry.builtin == builtin
                && entry.width == width
        })
        .map_or(BuiltinVerdict::Supported, |entry| entry.verdict)
}

/// Refuse dispatch when `builtin` is crash-class at `width` on this device.
///
/// Degraded and supported pairs pass through: degradation is a precision
/// caveat for tolerance selection, not a dispatch refusal.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] naming the operation, builtin,
/// and recorded cause when the table blocks the triple.
pub fn require_builtin(
    device: &WgpuDevice,
    builtin: MathBuiltin,
    width: CanaryWidth,
    operation: &'static str,
) -> Result<()> {
    match builtin_verdict(device, builtin, width) {
        BuiltinVerdict::Blocked { cause } => Err(HephaestusError::Unsupported {
            message: format!(
                "{operation}: WGSL {builtin:?} at {width:?} is driver-blocked: {cause}"
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

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Scan WGSL source for the canary builtins, honoring identifier boundaries.
///
/// `atanh(` must not match `tanh(`, and `log2(`/`log10(`/`lgamma(` must not
/// match `log(`; the template text is the ground truth of what the driver
/// will compile, so composite expressions are covered without a manual map.
#[must_use]
pub fn scan_builtins(source: &str) -> Vec<MathBuiltin> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    for (builtin, token) in TOKENS {
        let token = token.as_bytes();
        if bytes.len() < token.len() {
            continue;
        }
        let mut matched = false;
        for start in 0..=(bytes.len() - token.len()) {
            if bytes[start..start + token.len()] != *token {
                continue;
            }
            if start > 0 && is_ident_byte(bytes[start - 1]) {
                continue;
            }
            matched = true;
            break;
        }
        if matched {
            found.push(*builtin);
        }
    }
    found
}

/// Refuse dispatch when `source` uses a crash-class builtin at `width`.
///
/// Callers pass the template or composed WGSL they are about to compile; each
/// [`MathBuiltin`] found by [`scan_builtins`] goes through [`require_builtin`].
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
    Ok(())
}

/// Measured precision grade of one live probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionGrade {
    /// Within a few ulps of CPU truth at width precision.
    Exact,
    /// f64 computing at f32 grade (worst error near f32 epsilon).
    SingleGrade,
    /// Worse than f32 grade: a broken driver or a broken probe.
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
/// Crash-class pairs are refused from the recorded table before any shader is
/// built, so probing itself can never abort the process. f64 probing needs a
/// device with the `ShaderF64` feature; without it pipeline creation fails
/// with a typed validation error.
///
/// # Errors
///
/// Returns [`HephaestusError::Unsupported`] for table-blocked triples, and
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
mod tests {
    use super::*;

    #[test]
    fn width_of_covers_floats_only() {
        assert_eq!(width_of::<f32>(), Some(CanaryWidth::F32));
        assert_eq!(width_of::<f64>(), Some(CanaryWidth::F64));
        assert_eq!(width_of::<u32>(), None);
        assert_eq!(width_of::<i32>(), None);
    }

    #[test]
    fn scan_finds_builtin_calls_in_templates() {
        assert_eq!(scan_builtins("exp(x)"), vec![MathBuiltin::Exp]);
        assert_eq!(
            scan_builtins("log(x) * 0.43429448190325182"),
            vec![MathBuiltin::Log]
        );
        assert_eq!(scan_builtins("pow(a, b)"), vec![MathBuiltin::Pow]);
        assert_eq!(scan_builtins("tanh(x)"), vec![MathBuiltin::Tanh]);
        // Composite expressions carry their builtins with them.
        assert_eq!(
            scan_builtins(
                "0.5 * (x) * (1.0 + tanh(0.7978845608028654 * ((x) + 0.044715 * (x) * (x) * (x))))"
            ),
            vec![MathBuiltin::Tanh]
        );
        assert_eq!(scan_builtins("(exp(x) - 1.0)"), vec![MathBuiltin::Exp]);
    }

    #[test]
    fn scan_rejects_lookalike_prefixes() {
        // `atanh(` contains `tanh(`; the boundary check must reject it.
        assert!(scan_builtins("atanh(x)").is_empty());
        // Suffix builtins share the `log` stem but are distinct calls.
        assert!(scan_builtins("log2(x)").is_empty());
        assert!(scan_builtins("log10(x)").is_empty());
        assert!(scan_builtins("lgamma(x)").is_empty());
        assert!(scan_builtins("exp2(x)").is_empty());
        // Plain arithmetic carries no builtins.
        assert!(scan_builtins("(a) + (b)").is_empty());
        assert!(scan_builtins("select(0.0, 1.0, (x) > 0.0)").is_empty());
    }

    #[test]
    fn grades_cut_between_exact_single_and_poor() {
        let report = |width, err| PrecisionReport {
            builtin: MathBuiltin::Exp,
            width,
            samples: 8,
            max_relative_error: err,
        };
        assert_eq!(
            report(CanaryWidth::F64, 1e-15).grade(),
            PrecisionGrade::Exact
        );
        assert_eq!(
            report(CanaryWidth::F64, 5.96e-8).grade(),
            PrecisionGrade::SingleGrade
        );
        assert_eq!(report(CanaryWidth::F64, 1e-3).grade(), PrecisionGrade::Poor);
        assert_eq!(
            report(CanaryWidth::F32, 1e-7).grade(),
            PrecisionGrade::Exact
        );
        // f32 has no single-grade band: worse than exact is poor.
        assert_eq!(report(CanaryWidth::F32, 1e-4).grade(), PrecisionGrade::Poor);
    }
}
