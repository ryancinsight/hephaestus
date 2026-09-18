//! Expression and identity tests for the operation markers.

use super::*;
use crate::domain::dialect::{CudaC, HipC, Host, Wgsl};

#[test]
fn combine_and_identity_agree_per_dialect() {
    assert_eq!(<SumOp as CombineExpr<Wgsl>>::EXPR, "lhs + rhs");
    assert_eq!(<SumOp as CombineExpr<CudaC>>::EXPR, "lhs + rhs");
    assert_eq!(<SumOp as CombineExpr<HipC>>::EXPR, "lhs + rhs");
    assert_eq!(<ProdOp as CombineExpr<Wgsl>>::EXPR, "lhs * rhs");
    assert_eq!(<ProdOp as CombineExpr<CudaC>>::EXPR, "lhs * rhs");
    assert_eq!(<ProdOp as CombineExpr<HipC>>::EXPR, "lhs * rhs");
    assert_eq!(<AddOp as BinaryExpr<HipC>>::EXPR, "lhs + rhs");
    assert_eq!(<NegOp as UnaryExpr<HipC>>::EXPR, "-x");
    assert_eq!(
        <GeluTanhOp as UnaryExpr<Wgsl>>::EXPR,
        "0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))"
    );
    assert!(<SiluGradOp as UnaryExpr<CudaC>>::EXPR.contains("exp(-x)"));
    assert!(<SoftplusOp as UnaryExpr<HipC>>::EXPR.contains("exp(x)"));
    assert!(<MishOp as UnaryExpr<Wgsl>>::EXPR.contains("tanh(log(1.0 + exp(x)))"));
    assert!(<MishGradOp as UnaryExpr<CudaC>>::EXPR.contains("tanhf(logf(1.0f + expf(x)))"));
    assert_eq!(
        <EluOp as UnaryExpr<HipC>>::EXPR,
        "x >= 0.0f ? x : expf(x) - 1.0f"
    );
    assert_eq!(
        <EluGradOp as UnaryExpr<Wgsl>>::EXPR,
        "select(exp(x), 1.0, x >= 0.0)"
    );
    assert_eq!(
        <HardsigmoidOp as UnaryExpr<HipC>>::EXPR,
        "fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)"
    );
    assert_eq!(
        <HardsigmoidGradOp as UnaryExpr<HipC>>::EXPR,
        "(x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f"
    );
    assert_eq!(
        <HardswishOp as UnaryExpr<HipC>>::EXPR,
        "x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f"
    );
    assert_eq!(
        <HardswishGradOp as UnaryExpr<HipC>>::EXPR,
        "x >= 3.0f ? 1.0f : (x > -3.0f ? (2.0f * x + 3.0f) / 6.0f : 0.0f)"
    );
    assert_eq!(
        <SoftsignOp as UnaryExpr<HipC>>::EXPR,
        "x / (1.0f + fabsf(x))"
    );
    assert_eq!(
        <SoftsignGradOp as UnaryExpr<HipC>>::EXPR,
        "1.0f / ((1.0f + fabsf(x)) * (1.0f + fabsf(x)))"
    );
    assert_eq!(
        <Log10Op as UnaryExpr<Wgsl>>::EXPR,
        "log(x) * 0.43429448190325182f"
    );
    assert_eq!(<Expm1Op as UnaryExpr<CudaC>>::EXPR, "(exp(x) - 1.0f)");
    assert_eq!(<RoundOp as UnaryExpr<HipC>>::EXPR, "rint(x)");
    assert_eq!(<ErfOp as UnaryExpr<CudaC>>::EXPR, "erf(x)");
    assert_eq!(<ErfcOp as UnaryExpr<HipC>>::EXPR, "erfc(x)");
    assert_eq!(<LgammaOp as UnaryExpr<CudaC>>::EXPR, "lgamma(x)");
    assert_eq!(<LgammaOp as UnaryExpr<HipC>>::EXPR, "lgamma(x)");
    assert!(<ErfOp as UnaryExpr<Wgsl>>::EXPR.contains("1.061405429"));
    assert!(<ErfcOp as UnaryExpr<Wgsl>>::EXPR.starts_with("(1.0 - "));
    assert!(<LgammaOp as UnaryExpr<Wgsl>>::EXPR.contains("676.5203681218851"));
    assert!(<LgammaOp as UnaryExpr<Wgsl>>::EXPR.contains("abs(x) > 3.402823466e+38"));
    assert!(<GeluOp as UnaryExpr<Wgsl>>::EXPR.contains("0.7071067811865476"));
    assert!(<GeluGradOp as UnaryExpr<CudaC>>::EXPR.contains("erff(x *"));
    assert!(<GeluGradOp as UnaryExpr<HipC>>::EXPR.contains("expf(-0.5f * x * x)"));
    assert_eq!(
        <SignOp as UnaryExpr<Wgsl>>::EXPR,
        "select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0)"
    );
    assert_eq!(<f32 as IdentityToken<SumOp, Wgsl>>::TOKEN, "0.0");
    assert_eq!(<f32 as IdentityToken<SumOp, CudaC>>::TOKEN, "0.0f");
    assert_eq!(<f32 as IdentityToken<SumOp, HipC>>::TOKEN, "0.0f");
    assert_eq!(<f32 as IdentityToken<ProdOp, Wgsl>>::TOKEN, "1.0");
    assert_eq!(<f32 as IdentityToken<ProdOp, CudaC>>::TOKEN, "1.0f");
    assert_eq!(<f32 as IdentityToken<ProdOp, HipC>>::TOKEN, "1.0f");
    assert_eq!(<u32 as IdentityToken<ProdOp, Wgsl>>::TOKEN, "1u");
    assert_eq!(<u32 as IdentityToken<ProdOp, CudaC>>::TOKEN, "1u");
    assert_eq!(<u32 as IdentityToken<ProdOp, HipC>>::TOKEN, "1u");
    assert_eq!(<i32 as IdentityToken<ProdOp, Wgsl>>::TOKEN, "1");
    assert_eq!(<i32 as IdentityToken<ProdOp, CudaC>>::TOKEN, "1");
    assert_eq!(<i32 as IdentityToken<ProdOp, HipC>>::TOKEN, "1");
    assert_eq!(<f32 as OpIdentity<MinOp>>::IDENTITY, f32::MAX);
    assert_eq!(<f32 as OpIdentity<ProdOp>>::IDENTITY, 1.0);
    assert_eq!(<u32 as OpIdentity<ProdOp>>::IDENTITY, 1);
    assert_eq!(<i32 as OpIdentity<ProdOp>>::IDENTITY, 1);
    assert_eq!(<u32 as OpIdentity<MaxOp>>::IDENTITY, u32::MIN);
}

#[test]
fn comparisons_use_scalar_correct_mask_literals() {
    assert_eq!(
        <EqOp as TypedBinaryExpr<Wgsl, f32>>::EXPR,
        "select(0.0, 1.0, lhs == rhs)"
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<Wgsl, u32>>::EXPR,
        "select(0u, 1u, lhs == rhs)"
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<Wgsl, i32>>::EXPR,
        "select(0, 1, lhs == rhs)"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<CudaC, f32>>::EXPR,
        "lhs >= rhs ? 1.0f : 0.0f"
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs == rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <NeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs != rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <LtOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs < rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GtOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs > rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <LeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs <= rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<CudaC, f64>>::EXPR,
        "lhs >= rhs ? 1.0 : 0.0"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<CudaC, u32>>::EXPR,
        "lhs >= rhs ? 1u : 0u"
    );
    assert_eq!(
        <GeOp as TypedBinaryExpr<HipC, i32>>::EXPR,
        "lhs >= rhs ? 1 : 0"
    );
}

#[test]
fn consumer_defined_op_composes_with_the_vocabulary() {
    // A consumer-side fused op: implement the expression trait for a
    // local ZST in the targeted dialect — no substrate changes needed.
    #[derive(Clone, Copy, Debug, Default)]
    struct AffineClampOp;
    impl UnaryExpr<Wgsl> for AffineClampOp {
        const EXPR: &'static str = "clamp(x * 2.0 + 1.0, 0.0, 10.0)";
    }
    fn expr_of<Op: UnaryExpr<Wgsl>>() -> &'static str {
        Op::EXPR
    }
    assert_eq!(
        expr_of::<AffineClampOp>(),
        "clamp(x * 2.0 + 1.0, 0.0, 10.0)"
    );
}

/// Integer combines wrap (the WGSL kernel semantics); float combines are
/// IEEE; min/max keep `lhs` unless `rhs` is strictly smaller/larger.
#[test]
fn host_combines_apply_the_value_functions() {
    assert_eq!(
        <SumOp as CombineExpr<Host>>::value(i32::MAX, 1),
        Some(i32::MIN)
    );
    assert_eq!(
        <ProdOp as CombineExpr<Host>>::value(u32::MAX, 2),
        Some(u32::MAX - 1)
    );
    assert_eq!(
        <CumSumOp as CombineExpr<Host>>::value(1.5f32, 2.25),
        Some(3.75)
    );
    assert_eq!(<CumProdOp as CombineExpr<Host>>::value(-3i32, 4), Some(-12));
    assert_eq!(
        <MinOp as CombineExpr<Host>>::value(2.0f64, -1.0),
        Some(-1.0)
    );
    assert_eq!(<MaxOp as CombineExpr<Host>>::value(2u32, 7), Some(7));
    let kept = <MinOp as CombineExpr<Host>>::value(1.0f32, f32::NAN).expect("min has a value");
    assert_eq!(kept, 1.0, "a NaN rhs never displaces a held number");
    assert_eq!(<SumOp as CombineExpr<Host>>::EXPR, "host");
    assert_eq!(<f32 as IdentityToken<MinOp, Host>>::TOKEN, "host");
}

/// An operator implementing the host combine without a value function
/// reports `None`, which the host turns into a typed error; other dialects
/// never carry a value.
#[test]
fn a_host_combine_without_a_value_function_has_none() {
    #[derive(Clone, Copy)]
    struct Opaque;
    impl CombineExpr<Host> for Opaque {
        const EXPR: &'static str = "host";
    }
    assert_eq!(<Opaque as CombineExpr<Host>>::value(1.0f32, 2.0), None);
    assert_eq!(<SumOp as CombineExpr<Wgsl>>::value(1.0f32, 2.0), None);
}
