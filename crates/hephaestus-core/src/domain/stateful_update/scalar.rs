use eunomia::{FloatElement, Pod};

use super::super::device::DeviceFeature;
use super::super::dialect::{CudaC, DialectScalar, HipC, Host, Wgsl};

mod sealed {
    pub trait Sealed {}
}

/// Floating-point widths admitted by stateful parameter updates.
///
/// Sealed to `f32` and `f64`: every rule body is rendered once per width from
/// the same definition, so the admitted set is closed and each implementation
/// carries the width-specific facts the rule and backend layers consume. The
/// dialect supertraits make every admitted width representable on each
/// accelerator dialect, so a backend's generic dispatch needs no extra bound.
pub trait StatefulScalar:
    FloatElement
    + PartialOrd
    + core::fmt::Display
    + Pod
    + DialectScalar<Host>
    + DialectScalar<Wgsl>
    + DialectScalar<CudaC>
    + DialectScalar<HipC>
    + sealed::Sealed
{
    /// Literal spelling of `1.0` for this width in rendered kernel bodies.
    ///
    /// `f32` keeps the `f` suffix so C-family kernels evaluate in single
    /// precision; an unsuffixed literal would promote the expression to double.
    const ONE_LITERAL: &'static str;

    /// Device feature a backend must enable before launching at this width,
    /// or `None` when every supported device runs it.
    const REQUIRED_DEVICE_FEATURE: Option<DeviceFeature>;

    /// Raise `self` to a non-negative integer power with the width's native
    /// `powi`.
    ///
    /// Constructors use this for Adam bias corrections; delegating to the
    /// native method keeps the constructed bits identical to the pre-generic
    /// `f32`/`f64` implementations.
    #[must_use]
    fn integer_power(self, exponent: i32) -> Self;
}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}

impl StatefulScalar for f32 {
    const ONE_LITERAL: &'static str = "1.0f";
    const REQUIRED_DEVICE_FEATURE: Option<DeviceFeature> = None;

    fn integer_power(self, exponent: i32) -> Self {
        self.powi(exponent)
    }
}

impl StatefulScalar for f64 {
    const ONE_LITERAL: &'static str = "1.0";
    const REQUIRED_DEVICE_FEATURE: Option<DeviceFeature> = Some(DeviceFeature::ShaderF64);

    fn integer_power(self, exponent: i32) -> Self {
        self.powi(exponent)
    }
}
