use std::any::TypeId;

use hephaestus_core::{HephaestusError, Result};

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
pub(crate) fn restricted_width_undefined(device: &WgpuDevice, width: CanaryWidth) -> bool {
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
