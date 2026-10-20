//! Scalar vocabulary for provider-owned runtime-rank fusion.

use eunomia::Pod;

use crate::{
    CudaC, DialectScalar, HipC, IdentityToken, KernelDialect, MaxOp, MinOp, ProdOp, SumOp, Wgsl,
};

/// Scalar capabilities needed by provider-owned fusion in dialect `L`.
///
/// A type implements this seam only when its representation in dialect `L`
/// and all reduction identity tokens are available. This keeps unsupported
/// dialect types out of runtime kernel generation: a missing scalar is a
/// compile-time exclusion, never a runtime fallback.
pub trait FusionScalar<L: KernelDialect>: DialectScalar<L> + Pod + Send + Sync + 'static {
    /// Identity literal for addition.
    const ZERO: &'static str;
    /// Identity literal for multiplication.
    const ONE: &'static str;
    /// Lower-bound literal for maximum reduction.
    const LOWEST: &'static str;
    /// Upper-bound literal for minimum reduction.
    const HIGHEST: &'static str;
    /// Expression converting the runtime axis length for mean reduction.
    const MEAN_DIVISOR: &'static str;
}

macro_rules! impl_fusion_scalar {
    ($dialect:ty, $ty:ty, $divisor:literal) => {
        impl FusionScalar<$dialect> for $ty {
            const ZERO: &'static str = <Self as IdentityToken<SumOp, $dialect>>::TOKEN;
            const ONE: &'static str = <Self as IdentityToken<ProdOp, $dialect>>::TOKEN;
            const LOWEST: &'static str = <Self as IdentityToken<MaxOp, $dialect>>::TOKEN;
            const HIGHEST: &'static str = <Self as IdentityToken<MinOp, $dialect>>::TOKEN;
            const MEAN_DIVISOR: &'static str = $divisor;
        }
    };
}

impl_fusion_scalar!(Wgsl, f32, "f32(axis_length)");
impl_fusion_scalar!(Wgsl, f64, "f64(axis_length)");
impl_fusion_scalar!(Wgsl, i32, "i32(axis_length)");
impl_fusion_scalar!(Wgsl, u32, "u32(axis_length)");
impl_fusion_scalar!(Wgsl, eunomia::F16, "f16(axis_length)");

impl_fusion_scalar!(CudaC, f32, "static_cast<float>(axis_length)");
impl_fusion_scalar!(CudaC, i32, "static_cast<int>(axis_length)");
impl_fusion_scalar!(CudaC, u32, "static_cast<unsigned int>(axis_length)");

impl FusionScalar<CudaC> for f64 {
    const ZERO: &'static str = "0.0";
    const ONE: &'static str = "1.0";
    const LOWEST: &'static str = "-1.7976931348623157e+308";
    const HIGHEST: &'static str = "1.7976931348623157e+308";
    const MEAN_DIVISOR: &'static str = "static_cast<double>(axis_length)";
}

impl FusionScalar<CudaC> for eunomia::F16 {
    const ZERO: &'static str = <Self as IdentityToken<SumOp, CudaC>>::TOKEN;
    const ONE: &'static str = <Self as IdentityToken<ProdOp, CudaC>>::TOKEN;
    const LOWEST: &'static str = <Self as IdentityToken<MaxOp, CudaC>>::TOKEN;
    const HIGHEST: &'static str = <Self as IdentityToken<MinOp, CudaC>>::TOKEN;
    const MEAN_DIVISOR: &'static str = "__float2half(static_cast<float>(axis_length))";
}

impl FusionScalar<CudaC> for eunomia::Bf16 {
    const ZERO: &'static str = <Self as IdentityToken<SumOp, CudaC>>::TOKEN;
    const ONE: &'static str = <Self as IdentityToken<ProdOp, CudaC>>::TOKEN;
    const LOWEST: &'static str = <Self as IdentityToken<MaxOp, CudaC>>::TOKEN;
    const HIGHEST: &'static str = <Self as IdentityToken<MinOp, CudaC>>::TOKEN;
    const MEAN_DIVISOR: &'static str = "__float2bfloat16(static_cast<float>(axis_length))";
}

impl_fusion_scalar!(HipC, f32, "static_cast<float>(axis_length)");
impl_fusion_scalar!(HipC, i32, "static_cast<int>(axis_length)");
impl_fusion_scalar!(HipC, u32, "static_cast<unsigned int>(axis_length)");
impl_fusion_scalar!(HipC, f64, "static_cast<double>(axis_length)");
