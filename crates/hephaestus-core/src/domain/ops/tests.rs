//! Expression and identity tests for the operation markers.

use super::*;
use crate::domain::dialect::{CudaC, HipC, Host, Wgsl};
use eunomia::RealField;

mod double_double;

use double_double::DoubleDouble;

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

/// Integer `Add`/`Sub`/`Mul` wrap; `Div` returns the dividend on a zero
/// divisor or `MIN / -1` (WGSL semantics, ADR 0061 Decision 5); float `Div`
/// is IEEE.
#[test]
fn host_binary_value_functions_match_the_operator_definition() {
    assert_eq!(
        <AddOp as BinaryExpr<Host>>::value(i32::MAX, 1),
        Some(i32::MIN)
    );
    assert_eq!(
        <SubOp as BinaryExpr<Host>>::value(i32::MIN, 1),
        Some(i32::MAX)
    );
    assert_eq!(
        <MulOp as BinaryExpr<Host>>::value(u32::MAX, 2),
        Some(u32::MAX - 1)
    );
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(7i32, 0), Some(7));
    assert_eq!(
        <DivOp as BinaryExpr<Host>>::value(i32::MIN, -1),
        Some(i32::MIN)
    );
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(7.0f32, 2.0), Some(3.5));
    assert_eq!(<AddOp as BinaryExpr<Host>>::EXPR, "host");
}

/// ADR 0061 Verification plan: `f32` `Add` reaches `apply` through the
/// `apply_real` default (Add implements only `apply`), and `Pow` computes
/// through `apply_real` while its `apply` (the `NumericElement`-generic path
/// every integer scalar dispatches through) reports `None` for both `f32`
/// and `i32` alike — the host's per-scalar dispatch (`hephaestus-host`) is
/// what routes `f32` to `real_value` and `i32` to `value`.
#[test]
fn binary_dispatch_matches_the_adr_0061_verification_plan() {
    assert_eq!(
        <AddOp as BinaryExpr<Host>>::real_value(2.0f32, 3.0),
        Some(5.0)
    );
    assert_eq!(<PowOp as BinaryExpr<Host>>::value::<f32>(2.0, 3.0), None);
    assert_eq!(<PowOp as BinaryExpr<Host>>::value::<i32>(2, 3), None);
    assert_eq!(
        <PowOp as BinaryExpr<Host>>::real_value(2.0f32, 3.0),
        Some(8.0)
    );
}

/// Comparisons produce the type's `1`/`0` indicator, generic over every
/// `NumericElement` scalar; NaN follows `PartialOrd`'s unordered contract.
#[test]
fn host_typed_binary_value_functions_produce_indicators() {
    assert_eq!(
        <EqOp as TypedBinaryExpr<Host, f32>>::value(1.0, 1.0),
        Some(1.0)
    );
    assert_eq!(
        <EqOp as TypedBinaryExpr<Host, f32>>::value(1.0, 2.0),
        Some(0.0)
    );
    assert_eq!(
        <NeOp as TypedBinaryExpr<Host, f32>>::value(f32::NAN, f32::NAN),
        Some(1.0)
    );
    assert_eq!(
        <LtOp as TypedBinaryExpr<Host, f32>>::value(f32::NAN, 1.0),
        Some(0.0)
    );
    assert_eq!(<LtOp as TypedBinaryExpr<Host, i32>>::value(-5, 2), Some(1));
    assert_eq!(<GeOp as TypedBinaryExpr<Host, u32>>::value(3, 3), Some(1));
    assert_eq!(<EqOp as TypedBinaryExpr<Host, f32>>::EXPR, "host");
}

/// The value-function cancellation and rounding oracles ADR 0061 names,
/// exercised once for every shipped real scalar (`f32`, `f64`) rather than
/// per-type copies (standards: Generic Instantiation Coverage).
///
/// Tolerances are relative and derived from each function's own error
/// budget: `expm1`/`log1p`/`erfc` are libm-backed to a few ULP, so `1e-3`
/// relative is generous headroom above rounding noise while still catching a
/// cancellation collapse to zero (a collapse fails by 100%, not by ULPs).
fn assert_value_oracles_hold<T>()
where
    T: RealField + core::fmt::Debug,
{
    let tiny = T::from_f64(1e-10);
    let relative = |got: T, reference: f64| (got.to_f64() - reference).abs() / reference.abs();

    // exp(1e-10) - 1 would cancel to 0 at f32's ~1.2e-7 machine epsilon;
    // exp_m1 keeps the full relative value.
    assert!(
        relative(<Expm1Op as UnaryValue>::apply(tiny), 1e-10) < 1e-3,
        "expm1(1e-10) must retain relative precision"
    );
    // ln(1 + 1e-10) suffers the same cancellation; ln_1p does not.
    assert!(
        relative(<Log1pOp as UnaryValue>::apply(tiny), 1e-10) < 1e-3,
        "log1p(1e-10) must retain relative precision"
    );

    // softplus(-20) = ln(1+exp(-20)); the naive rendering underflows
    // ln(1+0) = 0 (ADR 0061 Decision 6), the value function must not.
    assert!(
        relative(
            <SoftplusOp as UnaryValue>::apply(T::from_f64(-20.0)),
            2.061_153_622e-9
        ) < 1e-3,
        "softplus(-20) must match the reference tail value"
    );
    // softplus(100) = 100 + a term far below 100's ULP; must not overflow
    // exp(100) into the sum.
    assert_eq!(
        <SoftplusOp as UnaryValue>::apply(T::from_f64(100.0)),
        T::from_f64(100.0),
        "softplus(100) must equal 100 exactly"
    );

    // gelu(-10) is a tiny but normal float; a naive 1+erf(-10/sqrt2)
    // cancellation would give exactly 0 (ADR 0061 Decision 6).
    let gelu_neg10 = <GeluOp as UnaryValue>::apply(T::from_f64(-10.0));
    assert!(
        gelu_neg10 != T::ZERO && gelu_neg10 < T::ZERO,
        "gelu(-10) must be a nonzero, negative-signed normal float, got {gelu_neg10:?}"
    );

    // silu(-89) sits in the WGSL rendering's measured underflow band
    // [-91.86, -88.72] (ADR 0061 Decision 6); the value function must not
    // reproduce it.
    let silu_neg89 = <SiluOp as UnaryValue>::apply(T::from_f64(-89.0));
    assert!(
        silu_neg89 != T::ZERO,
        "silu(-89) must not underflow to zero, got {silu_neg89:?}"
    );

    // RoundOp rounds ties to even, not away from zero.
    assert_eq!(
        <RoundOp as UnaryValue>::apply(T::from_f64(2.5)),
        T::from_f64(2.0)
    );
    assert_eq!(
        <RoundOp as UnaryValue>::apply(T::from_f64(3.5)),
        T::from_f64(4.0)
    );

    // SignOp is 0 at +-0 and NaN, not eunomia's signed `signum`.
    assert_eq!(<SignOp as UnaryValue>::apply(T::ZERO), T::ZERO);
    assert_eq!(<SignOp as UnaryValue>::apply(-T::ZERO), T::ZERO);
    assert_eq!(<SignOp as UnaryValue>::apply(T::NAN), T::ZERO);
}

#[test]
fn value_oracles_hold_for_every_shipped_real_scalar() {
    assert_value_oracles_hold::<f32>();
    assert_value_oracles_hold::<f64>();
}

/// The infinite-input regression set (ADR 0061 Decision 6). Every listed
/// formula forms `0 · ∞` or `∞ / ∞` at `x = ±∞` because its sigmoid/tanh
/// term saturates to exactly `0` or `1` there — `stable_sigmoid(-∞) == 0`
/// and `stable_sigmoid(+∞) == 1` bit-exactly, so e.g. `x * stable_sigmoid(x)`
/// at `x = -∞` is `-∞ · 0`. The value function must resolve each case to
/// its analytic limit instead of propagating `NaN`, for every shipped real
/// scalar (Generic Instantiation Coverage). Every assertion here fails
/// against the pre-fix code at commit ca3c28c, which returns `NaN` in every
/// case (confirmed by running this test before applying the fix).
fn assert_infinite_inputs_resolve_to_the_analytic_limit<T>()
where
    T: RealField + core::fmt::Debug,
{
    let neg_inf = T::neg_infinity();
    let pos_inf = T::infinity();

    // Silu, Mish, Gelu, GeluTanh: x -> x as x -> +-inf; only the negative
    // branch cancels (the naive product forms `-inf * 0`).
    let got = SiluOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "Silu(-inf) must be -0, got {got:?}"
    );
    assert_eq!(SiluOp::apply(pos_inf), pos_inf, "Silu(+inf) must be +inf");

    let got = MishOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "Mish(-inf) must be -0, got {got:?}"
    );
    assert_eq!(MishOp::apply(pos_inf), pos_inf, "Mish(+inf) must be +inf");

    let got = GeluOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "Gelu(-inf) must be -0, got {got:?}"
    );
    assert_eq!(GeluOp::apply(pos_inf), pos_inf, "Gelu(+inf) must be +inf");

    let got = GeluTanhOp::apply(neg_inf);
    assert!(
        got == T::ZERO && got.is_sign_negative(),
        "GeluTanh(-inf) must be -0, got {got:?}"
    );
    assert_eq!(
        GeluTanhOp::apply(pos_inf),
        pos_inf,
        "GeluTanh(+inf) must be +inf"
    );

    // Their gradients and GeluGrad: the analytic derivative limit is 1 at
    // +inf, 0 at -inf.
    assert_eq!(
        SiluGradOp::apply(pos_inf),
        T::ONE,
        "SiluGrad(+inf) must be 1"
    );
    assert_eq!(
        SiluGradOp::apply(neg_inf),
        T::ZERO,
        "SiluGrad(-inf) must be 0"
    );
    assert_eq!(
        MishGradOp::apply(pos_inf),
        T::ONE,
        "MishGrad(+inf) must be 1"
    );
    assert_eq!(
        MishGradOp::apply(neg_inf),
        T::ZERO,
        "MishGrad(-inf) must be 0"
    );
    assert_eq!(
        GeluGradOp::apply(pos_inf),
        T::ONE,
        "GeluGrad(+inf) must be 1"
    );
    assert_eq!(
        GeluGradOp::apply(neg_inf),
        T::ZERO,
        "GeluGrad(-inf) must be 0"
    );
    assert_eq!(
        GeluTanhGradOp::apply(pos_inf),
        T::ONE,
        "GeluTanhGrad(+inf) must be 1"
    );
    assert_eq!(
        GeluTanhGradOp::apply(neg_inf),
        T::ZERO,
        "GeluTanhGrad(-inf) must be 0"
    );

    // Softsign: x / (1 + |x|) forms inf/inf at +-inf; the limit is +-1.
    assert_eq!(
        SoftsignOp::apply(pos_inf),
        T::ONE,
        "Softsign(+inf) must be 1"
    );
    assert_eq!(
        SoftsignOp::apply(neg_inf),
        -T::ONE,
        "Softsign(-inf) must be -1"
    );
}

#[test]
fn infinite_inputs_resolve_to_the_analytic_limit_for_every_shipped_real_scalar() {
    assert_infinite_inputs_resolve_to_the_analytic_limit::<f32>();
    assert_infinite_inputs_resolve_to_the_analytic_limit::<f64>();
}

/// `GeluTanhGradOp` overflows on a *finite* input once `three * c1 * x * x`
/// itself exceeds the format's range (`|x| ≈ 5.04e19` in `f32`, `≈ 3.7e154`
/// in `f64`). By that magnitude the inner sigmoid has already saturated to
/// exactly `0`/`1` bit-exactly (its argument overflows to `±∞` through the
/// cubic term at a much smaller `|x|` — `c1 * x³` alone exceeds `f32::MAX`
/// past `|x| ≈ 1.97e13`), so the vanished `s(1-s)` factor multiplies an
/// overflowing `x²` (`0 · ∞ = NaN`) rather than the guard below returning
/// the already-saturated `s` directly. Fails against ca3c28c, which returns
/// `NaN` at every input here (confirmed before applying the fix).
#[test]
fn gelu_tanh_grad_resolves_the_overflowing_finite_tail() {
    assert_eq!(GeluTanhGradOp::apply(5.04e19_f32), 1.0);
    assert_eq!(GeluTanhGradOp::apply(-5.04e19_f32), 0.0);
    assert_eq!(GeluTanhGradOp::apply(3.7e154_f64), 1.0);
    assert_eq!(GeluTanhGradOp::apply(-3.7e154_f64), 0.0);
}

/// `HardswishOp` computed `x * clamp / 6`: for `x ≥ 3` the clamp saturates
/// to `6` and the true value is `x` exactly, but `x * 6` overflows before
/// the `/ 6` brings it back down once `x` exceeds `f32::MAX / 6 ≈ 5.67e37`
/// (`f64::MAX / 6 ≈ 3.0e307`). Dividing the clamp by `6` before multiplying
/// keeps every intermediate value at most `1`. Fails against ca3c28c, which
/// returns `inf` at both inputs (confirmed before applying the fix).
#[test]
fn hardswish_does_not_overflow_once_the_clamp_saturates() {
    assert_eq!(HardswishOp::apply(1e38_f32), 1e38_f32);
    assert_eq!(HardswishOp::apply(1e308_f64), 1e308_f64);
}

/// The NaN-propagation regression set: `ReluOp`/`ReluGradOp`,
/// `HardsigmoidOp`/`HardsigmoidGradOp`, and `HardswishGradOp` each branched
/// on a comparison (`x.max(0)`'s ignored-NaN contract, `x > 0`, `clamp`'s
/// same ignored-NaN contract, `x > -3 && x < 3`, a three-arm range test) —
/// every comparison against `NaN` is `false` under IEEE 754's unordered
/// semantics, so each fell through to an arm that returns a concrete
/// number (`0`, a clamp bound) instead of propagating `NaN`. An explicit
/// `is_nan` check now comes first in all five. Every assertion here fails
/// against the pre-fix code (confirmed by running this test before
/// applying the fix): all five silently returned `0` for a `NaN` input.
/// `HardswishOp` propagates `NaN` through its arithmetic because its
/// `x <= -3` short-circuit is `false` for `NaN`; the equivalent-looking
/// `!(x > -3)` guard is `true` for `NaN` and would return `-0` instead.
fn assert_nan_propagates_through_every_comparison_based_value_function<T>()
where
    T: RealField + core::fmt::Debug,
{
    let nan = T::NAN;
    assert!(ReluOp::apply(nan).is_nan(), "Relu(NaN) must be NaN");
    assert!(ReluGradOp::apply(nan).is_nan(), "ReluGrad(NaN) must be NaN");
    assert!(
        HardsigmoidOp::apply(nan).is_nan(),
        "Hardsigmoid(NaN) must be NaN"
    );
    assert!(
        HardsigmoidGradOp::apply(nan).is_nan(),
        "HardsigmoidGrad(NaN) must be NaN"
    );
    assert!(
        HardswishOp::apply(nan).is_nan(),
        "Hardswish(NaN) must be NaN"
    );
    assert!(
        HardswishGradOp::apply(nan).is_nan(),
        "HardswishGrad(NaN) must be NaN"
    );
}

#[test]
fn nan_propagates_through_every_comparison_based_value_function_for_every_shipped_real_scalar() {
    assert_nan_propagates_through_every_comparison_based_value_function::<f32>();
    assert_nan_propagates_through_every_comparison_based_value_function::<f64>();
}

/// `HardswishOp` at `x = -∞`: the clamp saturates to `0` there too, but the
/// literal `x * (0 / 6)` forms `-∞ · 0 = NaN` — a second-pass defect the
/// first fix (6ea4acd) left in place, since its `-∞`/`+∞` audit covered only
/// the sigmoid/tanh-based operators. Fails against 6ea4acd (and ca3c28c),
/// both of which return `NaN`.
#[test]
fn hardswish_resolves_negative_infinity_to_negative_zero() {
    let got32 = HardswishOp::apply(f32::NEG_INFINITY);
    assert!(
        got32 == 0.0 && got32.is_sign_negative(),
        "Hardswish(-inf) must be -0 in f32, got {got32:?}"
    );
    let got64 = HardswishOp::apply(f64::NEG_INFINITY);
    assert!(
        got64 == 0.0 && got64.is_sign_negative(),
        "Hardswish(-inf) must be -0 in f64, got {got64:?}"
    );
}

/// One accuracy case: an input, its reference value as the unevaluated
/// double-double sum `hi + lo`, and the largest admitted error in ULP of the
/// reference.
///
/// Every reference below is a double-double (about 106-bit) evaluation of the
/// operator's defining formula at the exact input value (an `f32` input is
/// widened exactly): `SiluGrad(x) = σ(x)·(1 + x·σ(−x))`;
/// `MishGrad(x) = tanh(sp) + x·sech²(sp)·σ(x)` with `sp = softplus(x)`;
/// `GeluTanhGrad(x) = σ(w) + k·σ(w)·σ(−w)` with `w = 2·c0·(x + c1·x³)` and
/// `k = 2·c0·x·(1 + 3·c1·x²)`, `c0 = 0.7978845608028654`, `c1 = 0.044715`;
/// `TanhGrad(y) = 1 − y²` with `y²` formed exactly by an error-free product.
/// Exponentials use a range-reduced Taylor series and `log1p` a Newton step
/// in double-double, so no reference shares a code path with eunomia's
/// `exp`/`ln_1p`/`tanh` or with the implementation's region choice.
///
/// Each case sits where the implementation's two forms (direct subtraction
/// from `1`, and the independent form) differ by at least one ULP, so it
/// fails when the switch routes the input to the wrong form. Each bound lies
/// strictly between the measured error of the form the implementation
/// selects and the measured error of the other form; both are recorded on
/// the case as `selected / other`.
struct AccuracyCase<T> {
    input: T,
    reference: (f64, f64),
    bound_ulp: f64,
}

/// Distance of `got` from `reference`, in ULP of the reference's leading
/// component for a format with `significand_bits` bits of precision (24 for
/// `f32`, 53 for `f64`).
fn ulps_from_reference(got: f64, reference: (f64, f64), significand_bits: u32) -> f64 {
    let (hi, lo) = reference;
    ((got - hi) - lo).abs() / ulp_of(hi, significand_bits)
}

/// ULP of the normal value `value` in a format with `significand_bits` bits
/// of precision: `2^(exponent(value) - significand_bits + 1)`.
fn ulp_of(value: f64, significand_bits: u32) -> f64 {
    let biased_exponent = i32::try_from((value.to_bits() >> 52) & 0x7ff)
        .expect("invariant: an 11-bit exponent field fits in i32");
    let precision =
        i32::try_from(significand_bits).expect("invariant: a significand width fits in i32");
    2f64.powi(biased_exponent - 1023 - (precision - 1))
}

fn assert_accuracy_cases<T>(
    operator: &str,
    apply: impl Fn(T) -> T,
    significand_bits: u32,
    cases: &[AccuracyCase<T>],
) where
    T: RealField + core::fmt::Debug,
{
    for case in cases {
        let got = apply(case.input);
        let error = ulps_from_reference(got.to_f64(), case.reference, significand_bits);
        assert!(
            error <= case.bound_ulp,
            "{operator}({:?}) = {got:?} is {error} ULP from the double-double \
             reference {:e} + {:e}, above the bound {}",
            case.input,
            case.reference.0,
            case.reference.1,
            case.bound_ulp
        );
    }
}

/// `SiluGradOp` switches from `1 - sigmoid(x)` to `sigmoid(-x)` above
/// `x = 2.63`. The cases below the switch kill an always-independent form and
/// any threshold at or below `2.40` (`f32`) / `2.55` (`f64`); the cases above
/// it kill an always-direct form and any threshold at or above `2.87` /
/// `2.90`, so a threshold moved by 10% either way fails; the tail cases are the original direct-form cancellation, where
/// `sigmoid(x)` has rounded so close to `1` that `1 - sigmoid(x)` keeps few
/// significant bits.
#[test]
fn silu_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "SiluGrad",
        SiluGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.115 / 1.885 ULP
            AccuracyCase {
                input: 2.405_141_8_f32,
                reference: (1.099_837_793_692_915_3, 6.976_311_581_976_778e-17),
                bound_ulp: 1.0,
            },
            // 0.017 / 2.017 ULP
            AccuracyCase {
                input: 2.867_271_7_f32,
                reference: (1.092_152_835_918_219_5, -6.123_557_629_831_417e-17),
                bound_ulp: 1.0,
            },
            // 0.215 / 7.785 ULP
            AccuracyCase {
                input: 16.64_f32,
                reference: (1.000_000_928_061_517_4, 1.188_321_631_270_538_1e-17),
                bound_ulp: 1.0,
            },
        ],
    );
    assert_accuracy_cases(
        "SiluGrad",
        SiluGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.091 / 1.909 ULP
            AccuracyCase {
                input: 2.552_028_415_350_948_f64,
                reference: (1.098_860_025_164_296_6, -2.016_099_943_032_254_3e-17),
                bound_ulp: 1.0,
            },
            // 0.009 / 2.009 ULP
            AccuracyCase {
                input: 2.903_298_488_057_376_7_f64,
                reference: (1.091_106_185_802_179_8, 1.901_338_674_739_636_7e-18),
                bound_ulp: 1.0,
            },
            // 0.987 / 18.013 ULP
            AccuracyCase {
                input: 36.73_f64,
                reference: (1.000_000_000_000_004, -2.907_402_465_278_531_4e-18),
                bound_ulp: 2.0,
            },
        ],
    );
}

/// `GeluTanhGradOp` switches from `1 - s` to `sigmoid(-w)` above
/// `w = 2z = 2.47`. The cases below the switch (`w ≈ 2.31`/`2.35`) kill an
/// always-independent form and any threshold at or below them; the cases
/// above it (`w ≈ 2.68`/`2.70`) kill an always-direct form and any threshold
/// at or above them, so a threshold moved by 10% either way fails; the tail cases are the original direct-form
/// cancellation, where `s` has rounded to `1`.
#[test]
fn gelu_tanh_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "GeluTanhGrad",
        GeluTanhGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.148 / 1.852 ULP
            AccuracyCase {
                input: 1.340_434_f32,
                reference: (1.127_676_469_325_747_5, 1.007_740_701_404_698_9e-16),
                bound_ulp: 1.0,
            },
            // 0.089 / 1.911 ULP
            AccuracyCase {
                input: 1.520_734_1_f32,
                reference: (1.127_006_281_717_678, -1.096_801_909_540_929_4e-18),
                bound_ulp: 1.0,
            },
            // 0.743 / 16.257 ULP
            AccuracyCase {
                input: 4.96_f32,
                reference: (1.000_001_995_905_308_4, -7.896_238_384_062_334e-17),
                bound_ulp: 2.0,
            },
        ],
    );
    assert_accuracy_cases(
        "GeluTanhGrad",
        GeluTanhGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.172 / 1.828 ULP
            AccuracyCase {
                input: 1.359_537_479_066_083_7_f64,
                reference: (1.128_251_526_320_643_6, 3.824_062_359_236_039_5e-17),
                bound_ulp: 1.0,
            },
            // 0.015 / 2.015 ULP
            AccuracyCase {
                input: 1.530_396_691_565_789_6_f64,
                reference: (1.126_630_199_140_305_6, -3.437_292_819_137_683e-18),
                bound_ulp: 1.0,
            },
            // 0.047 / 42.953 ULP
            AccuracyCase {
                input: 7.09_f64,
                reference: (1.000_000_000_000_009_5, -1.037_975_140_742_717_5e-17),
                bound_ulp: 1.0,
            },
        ],
    );
}

/// The root of `GeluTanhGrad` near `x = -0.7524614`.
const GELU_TANH_GRAD_ROOT: f64 = -0.752_461_4;

/// `GeluTanhGrad(x)` and its dominant term `s = σ(w)` in double-double,
/// from `σ(w) + k·σ(w)·σ(−w)` with `w = 2·c0·(x + c1·x³)` and
/// `k = 2·c0·x·(1 + 3·c1·x²)`.
fn gelu_tanh_grad_double_double(x: f64) -> (DoubleDouble, DoubleDouble) {
    let c0 = 0.797_884_560_802_865_4_f64;
    let c1 = 0.044715_f64;
    let one = DoubleDouble::from_f64(1.0);
    let xd = DoubleDouble::from_f64(x);
    let x2 = xd.mul(xd);
    let w =
        DoubleDouble::from_f64(2.0 * c0).mul(xd.add(DoubleDouble::from_f64(c1).mul(x2).mul(xd)));
    let e = w.neg().exp();
    let s = one.div(one.add(e));
    let one_minus_s = e.div(one.add(e));
    let k = DoubleDouble::from_f64(2.0 * c0)
        .mul(xd)
        .mul(one.add(DoubleDouble::from_f64(3.0 * c1).mul(x2)));
    (s.add(k.mul(s).mul(one_minus_s)), s)
}

/// Error of `got` from the double-double reference, in ULP of the dominant
/// term `s` for a format with `significand_bits` bits of precision.
fn gelu_tanh_grad_dominant_term_ulps(x: f64, got: f64, significand_bits: u32) -> f64 {
    let (reference, dominant) = gelu_tanh_grad_double_double(x);
    ((got - reference.hi) - reference.lo).abs() / ulp_of(dominant.hi, significand_bits)
}

/// Within `±0.05` of its root the two terms of `GeluTanhGrad` cancel, so an
/// error is measured in ULP of the dominant term `s = σ(2z)` (the result's
/// own ULP shrinks without bound at the root). Every 256th `f32` input of the
/// window and 4,096 evenly spaced `f64` inputs are checked against the
/// double-double reference.
///
/// Bound basis: the largest measured errors of the implementation over the
/// whole window are 4.681 dominant-term ULP over every `f32` input and 4.293
/// over 10,000,000 uniform `f64` samples (five seeds); `5.0` admits both with
/// margin and rejects an error of more than a third of an ULP beyond them.
#[test]
fn gelu_tanh_grad_stays_within_the_measured_bound_near_its_root() {
    const BOUND_DOMINANT_ULP: f64 = 5.0;
    let low = GELU_TANH_GRAD_ROOT - 0.05;
    let high = GELU_TANH_GRAD_ROOT + 0.05;
    // Negative `f32` bit patterns grow with magnitude.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the window bounds are representable to within one f32 rounding"
    )]
    let (first, last) = ((high as f32).to_bits(), (low as f32).to_bits());
    let mut worst32 = 0.0_f64;
    for bits in (first..=last).step_by(256) {
        let x = f32::from_bits(bits);
        let got = f64::from(GeluTanhGradOp::apply(x));
        let error = gelu_tanh_grad_dominant_term_ulps(f64::from(x), got, f32::MANTISSA_DIGITS);
        worst32 = worst32.max(error);
    }
    assert!(
        worst32 <= BOUND_DOMINANT_ULP,
        "f32 GeluTanhGrad near its root: {worst32} dominant-term ULP"
    );
    let samples = 4096_u32;
    let mut worst64 = 0.0_f64;
    for index in 0..samples {
        let x = low + (high - low) * (f64::from(index) + 0.5) / f64::from(samples);
        let got = GeluTanhGradOp::apply(x);
        let error = gelu_tanh_grad_dominant_term_ulps(x, got, f64::MANTISSA_DIGITS);
        worst64 = worst64.max(error);
    }
    assert!(
        worst64 <= BOUND_DOMINANT_ULP,
        "f64 GeluTanhGrad near its root: {worst64} dominant-term ULP"
    );
}

/// `GeluTanhGrad` by the original tanh form
/// `½·(1 + tanh z) + ½·x·(1 − tanh² z)·z′(x)`, evaluated in `f64` with
/// `std`'s `tanh` — a different algebraic form and library from the
/// implementation. Over the windows below `tanh z` stays below `0.93`, so
/// `1 − tanh² z` loses at most four bits of `f64`, far below an `f32` ULP.
fn gelu_tanh_grad_tanh_form(x: f64) -> f64 {
    let c0 = 0.797_884_560_802_865_4_f64;
    let c1 = 0.044715_f64;
    let z = c0 * (x + c1 * x * x * x);
    let t = z.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c0 * (1.0 + 3.0 * c1 * x * x)
}

/// Largest error, in ULP of the reference, of `GeluTanhGrad` over every
/// `f32` input in `[low, high]` (both positive).
fn gelu_tanh_grad_f32_window_maximum(low: f32, high: f32) -> f64 {
    let mut worst = 0.0_f64;
    for bits in low.to_bits()..=high.to_bits() {
        let x = f32::from_bits(bits);
        let reference = gelu_tanh_grad_tanh_form(f64::from(x));
        let got = f64::from(GeluTanhGradOp::apply(x));
        worst = worst.max(ulps_from_reference(
            got,
            (reference, 0.0),
            f32::MANTISSA_DIGITS,
        ));
    }
    worst
}

/// Each `GeluTanhGradOp` branch keeps the multiplication order of the
/// single-form evaluation it is measured against, and the other order raises
/// the `f32` window maximum (see the operator's documentation). Over every
/// `f32` input of each window below, the shipped order's maximum and the
/// other order's were measured against a double-double reference; each
/// bound lies between the two.
///
/// - `[0.515, 0.555]`, direct branch: 1.784 ULP factor by factor (shipped),
///   1.936 pre-grouped. Bound 1.86.
/// - `[1.625, 1.725]`, independent branch: 1.681 ULP pre-grouped (shipped),
///   1.774 factor by factor. Bound 1.73.
#[test]
fn gelu_tanh_grad_multiplication_orders_hold_their_measured_window_maxima() {
    let direct = gelu_tanh_grad_f32_window_maximum(0.515, 0.555);
    assert!(direct <= 1.86, "direct-branch window maximum {direct} ULP");
    let independent = gelu_tanh_grad_f32_window_maximum(1.625, 1.725);
    assert!(
        independent <= 1.73,
        "independent-branch window maximum {independent} ULP"
    );
}

/// `MishGradOp` switches from `1 - t²` to `sech²(sp)` above `sp = 1.55`. The
/// cases below the switch (`sp ≈ 1.03`/`1.11`, and `1.400`/`1.54999` just
/// inside it) kill an always-`sech²` form and any threshold below them,
/// including `1.55` moved down by 10% (`1.395`); the cases above it
/// (`sp ≈ 1.5503`/`1.605`, and `1.96`/`1.97`) kill an always-direct form
/// and any threshold at or above them, including `1.55` moved up by 10%
/// (`1.705`). The two forms differ by one ULP just inside the switch, so
/// those cases require the selected form to be correctly rounded (half an
/// ULP). The tail cases are the original direct-form cancellation, where
/// `t` has rounded to `1`.
#[test]
fn mish_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "MishGrad",
        MishGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.003 / 2.003 ULP
            AccuracyCase {
                input: 0.589_513_8_f32,
                reference: (9.261_161_090_871_325e-1, 2.060_666_313_517_767_4e-17),
                bound_ulp: 1.0,
            },
            // 0.009 / 1.009 ULP
            AccuracyCase {
                input: 1.116_696_7_f32,
                reference: (1.067_210_794_586_083_1, 6.519_837_544_120_851e-17),
                bound_ulp: 0.5,
            },
            // 0.015 / 1.015 ULP
            AccuracyCase {
                input: 1.311_765_7_f32,
                reference: (1.084_256_766_469_009_3, 2.719_817_198_156_410_7e-18),
                bound_ulp: 0.5,
            },
            // 0.447 / 1.553 ULP
            AccuracyCase {
                input: 1.810_998_3_f32,
                reference: (1.079_494_768_014_944_7, 6.340_200_556_247_342e-17),
                bound_ulp: 1.0,
            },
            // 0.261 / 4.739 ULP
            AccuracyCase {
                input: 9.01_f32,
                reference: (1.000_000_507_972_837_8, -1.054_899_121_256_476_9e-17),
                bound_ulp: 1.0,
            },
        ],
    );
    assert_accuracy_cases(
        "MishGrad",
        MishGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.016 / 2.016 ULP
            AccuracyCase {
                input: 0.707_785_371_182_024_1_f64,
                reference: (9.715_328_789_566_628e-1, -1.785_604_082_853_674_5e-18),
                bound_ulp: 1.0,
            },
            // 0.015 / 1.015 ULP
            AccuracyCase {
                input: 1.311_411_733_274_786_6_f64,
                reference: (1.084_238_829_570_115, 3.254_164_456_260_444_2e-18),
                bound_ulp: 0.5,
            },
            // 0.010 / 1.010 ULP
            AccuracyCase {
                input: 1.380_909_773_517_28_f64,
                reference: (1.086_984_743_270_022_8, -2.295_144_350_268_792_2e-18),
                bound_ulp: 0.5,
            },
            // 0.430 / 1.570 ULP
            AccuracyCase {
                input: 1.814_309_182_963_652_4_f64,
                reference: (1.079_334_820_511_441_7, -9.549_391_746_980_466e-17),
                bound_ulp: 1.0,
            },
            // 0.678 / 9.322 ULP
            AccuracyCase {
                input: 19.04_f64,
                reference: (1.000_000_000_000_002_2, -7.144_888_117_181_882e-17),
                bound_ulp: 2.0,
            },
        ],
    );
}

/// `TanhGradOp` switches from `1 - y*y` to `(1 - y) * (1 + y)` above
/// `|y| = 0.75`. The small-`|y|` cases kill an always-factored form; the
/// cases just above `√½` (`0.7071`) kill any threshold at or below them,
/// including the previous `½` and `0.75` moved down by 10% (`0.675`); the
/// cases at `0.778`/`0.8125` kill any threshold at or above them, including
/// `0.75` moved up by 10% (`0.825`), and those near `0.875`/`0.9` an
/// always-direct form; the cases near
/// `1` are the original direct-form cancellation. Where the two forms differ
/// by exactly one ULP the bound is half an ULP: the selected form must be
/// correctly rounded.
#[test]
fn tanh_grad_selects_the_more_accurate_form_on_each_side_of_its_crossover() {
    assert_accuracy_cases(
        "TanhGrad",
        TanhGradOp::apply,
        f32::MANTISSA_DIGITS,
        &[
            // 0.500 / 1.500 ULP
            AccuracyCase {
                input: 0.003_625_303_7_f32,
                reference: (9.999_868_571_727_95e-1, 4.157_915_331_481_909_5e-17),
                bound_ulp: 1.0,
            },
            // 0.000 / 1.000 ULP
            AccuracyCase {
                input: 0.500_000_06_f32,
                reference: (7.499_999_403_953_517e-1, 0.0),
                bound_ulp: 0.5,
            },
            // 0.015 / 1.015 ULP
            AccuracyCase {
                input: 0.707_112_13_f32,
                reference: (4.999_924_306_528_918e-1, 0.0),
                bound_ulp: 0.5,
            },
            // 0.000 / 1.000 ULP
            AccuracyCase {
                input: 0.812_500_24_f32,
                reference: (3.398_433_625_697_521e-1, 0.0),
                bound_ulp: 0.5,
            },
            // 0.000 / 2.000 ULP
            AccuracyCase {
                input: 0.874_999_9_f32,
                reference: (2.343_752_086_162_425e-1, 0.0),
                bound_ulp: 1.0,
            },
            // 0.299 / 989.299 ULP
            AccuracyCase {
                input: 0.99988_f32,
                reference: (2.399_539_036_694_875_4e-4, 0.0),
                bound_ulp: 1.0,
            },
        ],
    );
    assert_accuracy_cases(
        "TanhGrad",
        TanhGradOp::apply,
        f64::MANTISSA_DIGITS,
        &[
            // 0.433 / 1.433 ULP
            AccuracyCase {
                input: 0.005_575_415_809_682_238_f64,
                reference: (9.999_689_147_385_491e-1, 4.808_746_349_255_426_6e-17),
                bound_ulp: 1.0,
            },
            // 0.013 / 1.013 ULP
            AccuracyCase {
                input: 0.733_984_678_348_328_f64,
                reference: (4.612_664_919_499_014_4e-1, -7.084_623_958_970_427e-19),
                bound_ulp: 0.5,
            },
            // 0.000 / 1.000 ULP
            AccuracyCase {
                input: 0.777_782_521_737_601_f64,
                reference: (3.950_543_488_794_982_5e-1, 8.953_235_885_114_249e-25),
                bound_ulp: 0.5,
            },
            // 0.000 / 2.000 ULP
            AccuracyCase {
                input: 0.899_806_573_745_678_2_f64,
                reference: (1.903_481_298_440_634e-1, 2.693_694_929_135_220_4e-23),
                bound_ulp: 1.0,
            },
            // 0.015 / 27134340.015 ULP
            AccuracyCase {
                input: 0.999_999_993_3_f64,
                reference: (1.339_999_995_360_794_7e-8, -2.509_200_139_160_843_5e-26),
                bound_ulp: 1.0,
            },
        ],
    );
}

/// Integer `Div` edge cases and `Add`/`Mul` wraparound, generic over `i32`
/// and `u32` (the seam's shipped signed/unsigned integers).
#[test]
fn integer_binary_edge_cases_hold_for_every_shipped_integer_scalar() {
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(5i32, 0), Some(5));
    assert_eq!(
        <DivOp as BinaryExpr<Host>>::value(i32::MIN, -1),
        Some(i32::MIN)
    );
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(5u32, 0), Some(5));
    assert_eq!(<DivOp as BinaryExpr<Host>>::value(7i32, 2), Some(3));
    assert_eq!(
        <AddOp as BinaryExpr<Host>>::value(i32::MAX, 1),
        Some(i32::MIN)
    );
    assert_eq!(<AddOp as BinaryExpr<Host>>::value(u32::MAX, 1), Some(0u32));
    assert_eq!(
        <MulOp as BinaryExpr<Host>>::value(i32::MAX, 2),
        Some(i32::MAX.wrapping_mul(2))
    );
    assert_eq!(
        <MulOp as BinaryExpr<Host>>::value(u32::MAX, 2),
        Some(u32::MAX.wrapping_mul(2))
    );
}
