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
impl OpIdentity<SumOp> for f64 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<SumOp> for eunomia::F16 {
    const IDENTITY: Self = eunomia::F16::ZERO;
}
impl OpIdentity<SumOp> for eunomia::Bf16 {
    const IDENTITY: Self = eunomia::Bf16::ZERO;
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
impl OpIdentity<ProdOp> for f64 {
    const IDENTITY: Self = 1.0;
}
impl OpIdentity<ProdOp> for eunomia::F16 {
    const IDENTITY: Self = eunomia::F16::ONE;
}
impl OpIdentity<ProdOp> for eunomia::Bf16 {
    const IDENTITY: Self = eunomia::Bf16::ONE;
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
impl OpIdentity<MinOp> for f64 {
    const IDENTITY: Self = f64::MAX;
}
impl OpIdentity<MinOp> for eunomia::F16 {
    const IDENTITY: Self = eunomia::F16::from_bits(0x7BFF);
}
impl OpIdentity<MinOp> for eunomia::Bf16 {
    const IDENTITY: Self = eunomia::Bf16::from_bits(0x7F7F);
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
impl OpIdentity<MaxOp> for f64 {
    const IDENTITY: Self = f64::MIN;
}
impl OpIdentity<MaxOp> for eunomia::F16 {
    const IDENTITY: Self = eunomia::F16::from_bits(0xFBFF);
}
impl OpIdentity<MaxOp> for eunomia::Bf16 {
    const IDENTITY: Self = eunomia::Bf16::from_bits(0xFF7F);
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
impl OpIdentity<CumSumOp> for f64 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<CumSumOp> for eunomia::F16 {
    const IDENTITY: Self = eunomia::F16::ZERO;
}
impl OpIdentity<CumSumOp> for eunomia::Bf16 {
    const IDENTITY: Self = eunomia::Bf16::ZERO;
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
impl OpIdentity<CumProdOp> for f64 {
    const IDENTITY: Self = 1.0;
}
impl OpIdentity<CumProdOp> for eunomia::F16 {
    const IDENTITY: Self = eunomia::F16::ONE;
}
impl OpIdentity<CumProdOp> for eunomia::Bf16 {
    const IDENTITY: Self = eunomia::Bf16::ONE;
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
impl IdentityToken<SumOp, Wgsl> for f64 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<SumOp, Wgsl> for eunomia::F16 {
    const TOKEN: &'static str = "0.0";
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
impl IdentityToken<SumOp, CudaC> for f64 {
    const TOKEN: &'static str = "0.0";
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
impl IdentityToken<ProdOp, Wgsl> for f64 {
    const TOKEN: &'static str = "1.0";
}
impl IdentityToken<ProdOp, Wgsl> for eunomia::F16 {
    const TOKEN: &'static str = "1.0";
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
impl IdentityToken<ProdOp, CudaC> for f64 {
    const TOKEN: &'static str = "1.0";
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
impl IdentityToken<MinOp, Wgsl> for f64 {
    const TOKEN: &'static str = "1.7976931348623157e+308";
}
impl IdentityToken<MinOp, Wgsl> for eunomia::F16 {
    const TOKEN: &'static str = "65504.0";
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
impl IdentityToken<MinOp, CudaC> for f64 {
    const TOKEN: &'static str = "1.7976931348623157e+308";
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
impl IdentityToken<MaxOp, Wgsl> for f64 {
    const TOKEN: &'static str = "-1.7976931348623157e+308";
}
impl IdentityToken<MaxOp, Wgsl> for eunomia::F16 {
    const TOKEN: &'static str = "-65504.0";
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
impl IdentityToken<MaxOp, CudaC> for f64 {
    const TOKEN: &'static str = "-1.7976931348623157e+308";
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
impl IdentityToken<CumSumOp, Wgsl> for f64 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<CumSumOp, Wgsl> for eunomia::F16 {
    const TOKEN: &'static str = "0.0";
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
impl IdentityToken<CumSumOp, CudaC> for eunomia::F16 {
    const TOKEN: &'static str = "__float2half(0.0f)";
}
impl IdentityToken<CumSumOp, CudaC> for eunomia::Bf16 {
    const TOKEN: &'static str = "__float2bfloat16(0.0f)";
}
impl IdentityToken<CumSumOp, CudaC> for f64 {
    const TOKEN: &'static str = "0.0";
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
impl IdentityToken<CumProdOp, Wgsl> for f64 {
    const TOKEN: &'static str = "1.0";
}
impl IdentityToken<CumProdOp, Wgsl> for eunomia::F16 {
    const TOKEN: &'static str = "1.0";
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
impl IdentityToken<CumProdOp, CudaC> for eunomia::F16 {
    const TOKEN: &'static str = "__float2half(1.0f)";
}
impl IdentityToken<CumProdOp, CudaC> for eunomia::Bf16 {
    const TOKEN: &'static str = "__float2bfloat16(1.0f)";
}
impl IdentityToken<CumProdOp, CudaC> for f64 {
    const TOKEN: &'static str = "1.0";
}

/// The host renders no literals: any scalar with a host-side identity for
/// `Op` has the fixed host token.
impl<Op, T: OpIdentity<Op> + DialectScalar<Host>> IdentityToken<Op, Host> for T {
    const TOKEN: &'static str = "host";
}

macro_rules! impl_hip_identity_tokens {
    ($(($op:ty, $f32_token:literal, $u32_token:literal, $i32_token:literal, $f64_token:literal)),+ $(,)?) => {
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
            impl IdentityToken<$op, HipC> for f64 {
                const TOKEN: &'static str = $f64_token;
            }
        )+
    };
}

impl_hip_identity_tokens!(
    (SumOp, "0.0f", "0u", "0", "0.0"),
    (ProdOp, "1.0f", "1u", "1", "1.0"),
    (
        MinOp,
        "3.402823466e+38f",
        "4294967295u",
        "2147483647",
        "1.7976931348623157e+308"
    ),
    (
        MaxOp,
        "-3.402823466e+38f",
        "0u",
        "-2147483648",
        "-1.7976931348623157e+308"
    ),
    (CumSumOp, "0.0f", "0u", "0", "0.0"),
    (CumProdOp, "1.0f", "1u", "1", "1.0"),
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_cum_tokens_match_half_constructors() {
        assert_eq!(
            <eunomia::F16 as IdentityToken<CumSumOp, CudaC>>::TOKEN,
            "__float2half(0.0f)"
        );
        assert_eq!(
            <eunomia::Bf16 as IdentityToken<CumSumOp, CudaC>>::TOKEN,
            "__float2bfloat16(0.0f)"
        );
        assert_eq!(
            <eunomia::F16 as IdentityToken<CumProdOp, CudaC>>::TOKEN,
            "__float2half(1.0f)"
        );
        assert_eq!(
            <eunomia::Bf16 as IdentityToken<CumProdOp, CudaC>>::TOKEN,
            "__float2bfloat16(1.0f)"
        );
    }

    #[test]
    fn halves_host_identities_match_zero_one_max() {
        assert_eq!(
            <eunomia::F16 as OpIdentity<SumOp>>::IDENTITY,
            eunomia::F16::ZERO
        );
        assert_eq!(
            <eunomia::Bf16 as OpIdentity<ProdOp>>::IDENTITY,
            eunomia::Bf16::ONE
        );
        assert_eq!(
            <eunomia::F16 as OpIdentity<MinOp>>::IDENTITY,
            eunomia::F16::from_bits(0x7BFF)
        );
        assert_eq!(
            <eunomia::Bf16 as OpIdentity<MaxOp>>::IDENTITY,
            eunomia::Bf16::from_bits(0xFF7F)
        );
    }

    #[test]
    fn wgsl_f16_tokens_are_abstract_float_literals() {
        // Identity tokens initialize or assign into `f16`-typed places, so
        // AbstractFloat literals convert implicitly; no `h` suffix needed.
        assert_eq!(<eunomia::F16 as IdentityToken<SumOp, Wgsl>>::TOKEN, "0.0");
        assert_eq!(<eunomia::F16 as IdentityToken<ProdOp, Wgsl>>::TOKEN, "1.0");
        assert_eq!(
            <eunomia::F16 as IdentityToken<MinOp, Wgsl>>::TOKEN,
            "65504.0"
        );
        assert_eq!(
            <eunomia::F16 as IdentityToken<MaxOp, Wgsl>>::TOKEN,
            "-65504.0"
        );
        assert_eq!(
            <eunomia::F16 as IdentityToken<CumSumOp, Wgsl>>::TOKEN,
            "0.0"
        );
        assert_eq!(
            <eunomia::F16 as IdentityToken<CumProdOp, Wgsl>>::TOKEN,
            "1.0"
        );
    }

    #[test]
    fn existing_tokens_unchanged() {
        assert_eq!(<f32 as IdentityToken<SumOp, CudaC>>::TOKEN, "0.0f");
        assert_eq!(
            <f64 as IdentityToken<MinOp, CudaC>>::TOKEN,
            "1.7976931348623157e+308"
        );
    }
}
