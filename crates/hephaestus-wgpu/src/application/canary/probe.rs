use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, Wgsl};

use crate::application::elementwise::encode_elementwise;
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::infrastructure::device::WgpuDevice;

use super::verdict::BuiltinVerdict;
use super::verdict::CanaryWidth;
use super::verdict::MathBuiltin;
use super::verdict::builtin_verdict;

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
