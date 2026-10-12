use hephaestus_core::{HephaestusError, Result};

use super::widths::{CanaryWidth, MathBuiltin};
use crate::infrastructure::device::WgpuDevice;

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
/// Returns [`HephaestusError::Unsupported`](hephaestus_core::HephaestusError::Unsupported) naming the operation, builtin,
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
