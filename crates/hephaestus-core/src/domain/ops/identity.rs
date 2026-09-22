//! Identity elements of the combine markers, host-side and per dialect.

use super::{CumProdOp, CumSumOp, MaxOp, MinOp, ProdOp, SumOp};
use crate::domain::dialect::{CudaC, DialectScalar, HipC, Host, KernelDialect, Wgsl};
use eunomia::Pod;

/// Host-side identity element of op `Op` for this scalar (dialect-free).
pub trait OpIdentity<Op>: Pod {
    /// The identity value (e.g. `0` for sum, `T::MAX` for min).
    const IDENTITY: Self;
}

/// Shader literal token of op `Op`'s identity for this scalar in dialect `L`.
pub trait IdentityToken<Op, L: KernelDialect>: DialectScalar<L> {
    /// The dialect literal (e.g. `"0.0"` in WGSL, `"0.0f"` in CUDA C++).
    const TOKEN: &'static str;
}

// ── Identities ───────────────────────────────────────────────────────────
// Host values are dialect-free; literal tokens differ per dialect (WGSL has
// no `f` suffix, CUDA C++ float literals carry one so arithmetic stays in
// `float` rather than promoting to `double`).

impl OpIdentity<SumOp> for f32 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<SumOp> for u32 {
    const IDENTITY: Self = 0;
}
impl OpIdentity<SumOp> for i32 {
    const IDENTITY: Self = 0;
}

impl OpIdentity<ProdOp> for f32 {
    const IDENTITY: Self = 1.0;
}
impl OpIdentity<ProdOp> for u32 {
    const IDENTITY: Self = 1;
}
impl OpIdentity<ProdOp> for i32 {
    const IDENTITY: Self = 1;
}

impl OpIdentity<MinOp> for f32 {
    const IDENTITY: Self = f32::MAX;
}
impl OpIdentity<MinOp> for u32 {
    const IDENTITY: Self = u32::MAX;
}
impl OpIdentity<MinOp> for i32 {
    const IDENTITY: Self = i32::MAX;
}

impl OpIdentity<MaxOp> for f32 {
    const IDENTITY: Self = f32::MIN;
}
impl OpIdentity<MaxOp> for u32 {
    const IDENTITY: Self = u32::MIN;
}
impl OpIdentity<MaxOp> for i32 {
    const IDENTITY: Self = i32::MIN;
}

impl OpIdentity<CumSumOp> for f32 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<CumSumOp> for u32 {
    const IDENTITY: Self = 0;
}
impl OpIdentity<CumSumOp> for i32 {
    const IDENTITY: Self = 0;
}

impl OpIdentity<CumProdOp> for f32 {
    const IDENTITY: Self = 1.0;
}
impl OpIdentity<CumProdOp> for u32 {
    const IDENTITY: Self = 1;
}
impl OpIdentity<CumProdOp> for i32 {
    const IDENTITY: Self = 1;
}

impl IdentityToken<SumOp, Wgsl> for f32 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<SumOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<SumOp, Wgsl> for i32 {
    const TOKEN: &'static str = "0";
}
impl IdentityToken<SumOp, CudaC> for f32 {
    const TOKEN: &'static str = "0.0f";
}
impl IdentityToken<SumOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<SumOp, CudaC> for i32 {
    const TOKEN: &'static str = "0";
}
impl IdentityToken<SumOp, CudaC> for eunomia::F16 {
    const TOKEN: &'static str = "__float2half(0.0f)";
}
impl IdentityToken<SumOp, CudaC> for eunomia::Bf16 {
    const TOKEN: &'static str = "__float2bfloat16(0.0f)";
}

impl IdentityToken<ProdOp, Wgsl> for f32 {
    const TOKEN: &'static str = "1.0";
}
impl IdentityToken<ProdOp, Wgsl> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<ProdOp, Wgsl> for i32 {
    const TOKEN: &'static str = "1";
}
impl IdentityToken<ProdOp, CudaC> for f32 {
    const TOKEN: &'static str = "1.0f";
}
impl IdentityToken<ProdOp, CudaC> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<ProdOp, CudaC> for i32 {
    const TOKEN: &'static str = "1";
}
impl IdentityToken<ProdOp, CudaC> for eunomia::F16 {
    const TOKEN: &'static str = "__float2half(1.0f)";
}
impl IdentityToken<ProdOp, CudaC> for eunomia::Bf16 {
    const TOKEN: &'static str = "__float2bfloat16(1.0f)";
}

impl IdentityToken<MinOp, Wgsl> for f32 {
    const TOKEN: &'static str = "3.402823466e+38";
}
impl IdentityToken<MinOp, Wgsl> for u32 {
    const TOKEN: &'static str = "4294967295u";
}
impl IdentityToken<MinOp, Wgsl> for i32 {
    const TOKEN: &'static str = "2147483647";
}
impl IdentityToken<MinOp, CudaC> for f32 {
    const TOKEN: &'static str = "3.402823466e+38f";
}
impl IdentityToken<MinOp, CudaC> for u32 {
    const TOKEN: &'static str = "4294967295u";
}
impl IdentityToken<MinOp, CudaC> for i32 {
    const TOKEN: &'static str = "2147483647";
}
impl IdentityToken<MinOp, CudaC> for eunomia::F16 {
    const TOKEN: &'static str = "__float2half(65504.0f)";
}
impl IdentityToken<MinOp, CudaC> for eunomia::Bf16 {
    const TOKEN: &'static str = "__float2bfloat16(3.38953139e+38f)";
}

impl IdentityToken<MaxOp, Wgsl> for f32 {
    const TOKEN: &'static str = "-3.402823466e+38";
}
impl IdentityToken<MaxOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<MaxOp, Wgsl> for i32 {
    const TOKEN: &'static str = "-2147483648";
}
impl IdentityToken<MaxOp, CudaC> for f32 {
    const TOKEN: &'static str = "-3.402823466e+38f";
}
impl IdentityToken<MaxOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<MaxOp, CudaC> for i32 {
    const TOKEN: &'static str = "-2147483648";
}
impl IdentityToken<MaxOp, CudaC> for eunomia::F16 {
    const TOKEN: &'static str = "__float2half(-65504.0f)";
}
impl IdentityToken<MaxOp, CudaC> for eunomia::Bf16 {
    const TOKEN: &'static str = "__float2bfloat16(-3.38953139e+38f)";
}

impl IdentityToken<CumSumOp, Wgsl> for f32 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<CumSumOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<CumSumOp, Wgsl> for i32 {
    const TOKEN: &'static str = "0";
}
impl IdentityToken<CumSumOp, CudaC> for f32 {
    const TOKEN: &'static str = "0.0f";
}
impl IdentityToken<CumSumOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<CumSumOp, CudaC> for i32 {
    const TOKEN: &'static str = "0";
}

impl IdentityToken<CumProdOp, Wgsl> for f32 {
    const TOKEN: &'static str = "1.0";
}
impl IdentityToken<CumProdOp, Wgsl> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<CumProdOp, Wgsl> for i32 {
    const TOKEN: &'static str = "1";
}
impl IdentityToken<CumProdOp, CudaC> for f32 {
    const TOKEN: &'static str = "1.0f";
}
impl IdentityToken<CumProdOp, CudaC> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<CumProdOp, CudaC> for i32 {
    const TOKEN: &'static str = "1";
}

/// The host renders no literals: any scalar with a host-side identity for
/// `Op` has the fixed host token.
impl<Op, T: OpIdentity<Op> + DialectScalar<Host>> IdentityToken<Op, Host> for T {
    const TOKEN: &'static str = "host";
}

macro_rules! impl_hip_identity_tokens {
    ($(($op:ty, $f32_token:literal, $u32_token:literal, $i32_token:literal)),+ $(,)?) => {
        $(
            impl IdentityToken<$op, HipC> for f32 {
                const TOKEN: &'static str = $f32_token;
            }
            impl IdentityToken<$op, HipC> for u32 {
                const TOKEN: &'static str = $u32_token;
            }
            impl IdentityToken<$op, HipC> for i32 {
                const TOKEN: &'static str = $i32_token;
            }
        )+
    };
}

impl_hip_identity_tokens!(
    (SumOp, "0.0f", "0u", "0"),
    (ProdOp, "1.0f", "1u", "1"),
    (MinOp, "3.402823466e+38f", "4294967295u", "2147483647"),
    (MaxOp, "-3.402823466e+38f", "0u", "-2147483648"),
    (CumSumOp, "0.0f", "0u", "0"),
    (CumProdOp, "1.0f", "1u", "1"),
);
