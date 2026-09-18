//! Expression and identity tests for the operation markers.

use super::*;
use crate::domain::dialect::{CudaC, HipC, Host, Wgsl};
use eunomia::RealField;

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

/// `n` ULP of `f32`, as a relative bound: `f32::EPSILON` (`2^-23`) is
/// exactly one ULP at magnitudes in `[1, 2)` and within a factor of 2 at any
/// other normal magnitude — the standard "within `n` ULP" engineering
/// bound. Used to judge an `f32` result against a much higher-precision
/// `f64` reference (eunomia's f64 `exp`/`tanh` are libm-backed to a few ULP
/// of `f64`, roughly `2^29` times finer than an `f32` ULP, so evaluating the
/// same fixed formula in `f64` at the promoted input is authoritative at
/// this resolution).
fn ulp_bound_f32(n_ulp: f64, reference: f64) -> f64 {
    n_ulp * f64::from(f32::EPSILON) * reference.abs()
}

/// `n` ULP of `f64`, as a relative bound (`f64::EPSILON = 2^-52`), used
/// below to judge a result against an *independently derived* asymptotic
/// reference — an analytic series expansion, not the code path under test —
/// so the check is a real cross-verification (standards: Cross-Verification)
/// rather than comparing the implementation against itself.
fn ulp_bound_f64(n_ulp: f64, reference: f64) -> f64 {
    n_ulp * f64::EPSILON * reference.abs()
}

/// `SiluGradOp` computed `1 - sigmoid(x)` by direct subtraction, which loses
/// precision once `sigmoid(x)` rounds close to `1`: ca3c28c gives 1.0000019
/// at `x = 16.64` (f32) against the measured reference 1.00000093 (8.2 ULP
/// of f32), and at `x = 36.73` (f64) collapses to exactly `1.0` (19 ULP of
/// f64) because `sigmoid(36.73)` itself rounds to bit-exact `1.0`, so
/// `1 - sigmoid(x)` is exactly `0` and the whole correction term vanishes.
/// `stable_sigmoid(-x)` computes the identical quantity via
/// `1 - sigmoid(x) = sigmoid(-x)`, never subtracting two nearly-equal
/// numbers.
#[test]
fn silu_grad_matches_the_stable_one_minus_sigmoid_reference() {
    let reference_f32 = SiluGradOp::apply(f64::from(16.64_f32));
    let got_f32 = f64::from(SiluGradOp::apply(16.64_f32));
    assert!(
        (got_f32 - reference_f32).abs() <= ulp_bound_f32(4.0, reference_f32),
        "SiluGrad(16.64f32) = {got_f32}, reference {reference_f32}"
    );

    // Independent f64 reference: SiluGrad(x) = sigmoid(x) + x*sigmoid(x)*(1
    // - sigmoid(x)). Writing q = exp(-x), sigmoid(x) = 1/(1+q) = 1 - q +
    // O(q²), so SiluGrad(x) = 1 + q*(x - 1) + O(q²). At x = 36.73,
    // q ≈ 1.1e-16, so the dropped O(q²) term is ~1e-32 relative to the
    // O(q) ≈ 4e-15 leading correction — far below f64's own 2.2e-16
    // epsilon, making this asymptotic expansion exact to full f64
    // precision. It is derived independently of `stable_sigmoid`, so
    // matching it is a real cross-check, not self-comparison.
    let x = 36.73_f64;
    let reference_f64 = 1.0 + (x - 1.0) * (-x).exp();
    let got_f64 = SiluGradOp::apply(x);
    assert!(
        (got_f64 - reference_f64).abs() <= ulp_bound_f64(4.0, reference_f64),
        "SiluGrad(36.73f64) = {got_f64}, reference {reference_f64}"
    );
}

/// `MishGradOp` computed `1 - t * t` by direct subtraction, which loses
/// precision once `t = tanh(softplus(x))` rounds close to `1`: ca3c28c
/// gives 1.0000011 at `x = 9.01` (f32) against the measured reference
/// 1.00000051 (4.75 ULP of f32), and at `x = 19.04` (f64) the same collapse
/// to exactly `1.0` (10 ULP of f64). `sech²(sp) = 4u / (1+u)²`,
/// `u = exp(-2·sp)` (the identity `1 - tanh(sp)² = sech²(sp)`) computes the
/// identical quantity without subtracting from `1`.
#[test]
fn mish_grad_matches_the_stable_sech_squared_reference() {
    let reference_f32 = MishGradOp::apply(f64::from(9.01_f32));
    let got_f32 = f64::from(MishGradOp::apply(9.01_f32));
    assert!(
        (got_f32 - reference_f32).abs() <= ulp_bound_f32(4.0, reference_f32),
        "MishGrad(9.01f32) = {got_f32}, reference {reference_f32}"
    );

    // Independent f64 reference: for large x, softplus(x) = x + O(exp(-x))
    // so t = tanh(softplus(x)) = 1 - 2r + O(r²) with r = exp(-2x), and
    // sigmoid(x) = 1 - exp(-x) + O(exp(-2x)). MishGrad(x) = t +
    // x*(1-t²)*sigmoid(x) = (1-2r) + x*4r*(1-exp(-x)) + O(r²) =
    // 1 + r*(4x - 2) + O(r·exp(-x), r²), both dropped terms ~1e-33 or
    // smaller at x = 19.04 (r ≈ 2.9e-17) — exact to full f64 precision, and
    // independent of the `sech²`/`stable_sigmoid` code path under test.
    let x = 19.04_f64;
    let r = (-2.0 * x).exp();
    let reference_f64 = 1.0 + (4.0 * x - 2.0) * r;
    let got_f64 = MishGradOp::apply(x);
    assert!(
        (got_f64 - reference_f64).abs() <= ulp_bound_f64(4.0, reference_f64),
        "MishGrad(19.04f64) = {got_f64}, reference {reference_f64}"
    );
}

/// `GeluTanhGradOp` computed `1 - s` by direct subtraction, which loses
/// precision once `s = sigmoid(2z)` rounds close to `1`: ca3c28c gives `1.0`
/// at `x = 4.96` (f32) against the measured reference 1.00000197 (16.5 ULP
/// of f32), and at `x = 7.09` (f64) the same collapse to exactly `1.0`
/// (44 ULP of f64). `stable_sigmoid(-2z)` computes the identical quantity
/// via `1 - sigmoid(w) = sigmoid(-w)`, never subtracting from `1`.
#[test]
fn gelu_tanh_grad_matches_the_stable_one_minus_sigmoid_reference() {
    let reference_f32 = GeluTanhGradOp::apply(f64::from(4.96_f32));
    let got_f32 = f64::from(GeluTanhGradOp::apply(4.96_f32));
    assert!(
        (got_f32 - reference_f32).abs() <= ulp_bound_f32(4.0, reference_f32),
        "GeluTanhGrad(4.96f32) = {got_f32}, reference {reference_f32}"
    );

    // Independent f64 reference, re-deriving `z` and the coefficient `k`
    // directly from the operator's own closed form (never calling
    // `stable_sigmoid`): with w = 2z, s = sigmoid(w) = 1 - exp(-w) +
    // O(exp(-2w)), and the derivative's exact rational form is
    // s + 2x·s(1-s)·c0·(1+3c1x²) = s + k·s(1-s) for k = 2x·c0·(1+3c1x²).
    // s(1-s) = exp(-w) + O(exp(-2w)), so GeluTanhGrad(x) =
    // 1 + (k - 1)*exp(-w) + O(exp(-2w)); at x = 7.09, w ≈ 36.8, so the
    // dropped O(exp(-2w)) term is astronomically smaller than f64 epsilon.
    let x = 7.09_f64;
    let c0 = 0.797_884_560_802_865_4_f64;
    let c1 = 0.044715_f64;
    let z = c0 * (x + c1 * x * x * x);
    let w = 2.0 * z;
    let k = 2.0 * x * c0 * (1.0 + 3.0 * c1 * x * x);
    let reference_f64 = 1.0 + (k - 1.0) * (-w).exp();
    let got_f64 = GeluTanhGradOp::apply(x);
    assert!(
        (got_f64 - reference_f64).abs() <= ulp_bound_f64(4.0, reference_f64),
        "GeluTanhGrad(7.09f64) = {got_f64}, reference {reference_f64}"
    );
}

/// `TanhGradOp` computed `1 - y * y`, which squares `y` before subtracting —
/// doubling the rounding error already present in `y` once it rounds close
/// to `1`. ca3c28c gives 2.44141e-4 at `y = 0.99988` (f32) against the
/// measured reference 2.44126e-4 (~1024 ULP of f32), and the same
/// cancellation is far worse at `y = 1 - 6.7e-9` (f64, ~2.7e7 ULP of f64) —
/// the closer `y` sits to `1`, the more severe. `(1 - y) * (1 + y)` keeps
/// each factor's own rounding separate instead of compounding it in `y * y`.
#[test]
fn tanh_grad_matches_the_factored_difference_of_squares_reference() {
    let y_f32 = 0.99988_f32;
    let reference_f32 = TanhGradOp::apply(f64::from(y_f32));
    let got_f32 = f64::from(TanhGradOp::apply(y_f32));
    assert!(
        (got_f32 - reference_f32).abs() <= ulp_bound_f32(4.0, reference_f32),
        "TanhGrad(0.99988f32) = {got_f32}, reference {reference_f32}"
    );

    // Independent f64 reference: with y = 1 - delta, 1 - y² expands exactly
    // (no series truncation) as delta*(2 - delta) — algebraically identical
    // to `(1-y)*(1+y)` but computed from `delta` directly, never forming
    // `y*y`. `delta` is recovered from the actual stored `y_f64` (`1 - y`,
    // exact by Sterbenz's lemma since `y_f64` and `1.0` are within a factor
    // of 2) rather than reused from the `6.7e-9` literal, so the reference
    // is not contaminated by the literal's own rounding when `y_f64` was
    // first formed (`1.0 - 6.7e-9` itself rounds to the nearest `f64`,
    // ~1e-16 absolute); this isolates exactly the property under test — is
    // `1 - y²` computed accurately given the `y` that is actually stored —
    // without also asserting precision the input's own representation
    // cannot supply.
    let y_f64 = 1.0 - 6.7e-9;
    let delta = 1.0 - y_f64;
    let reference_f64 = delta * (2.0 - delta);
    let got_f64 = TanhGradOp::apply(y_f64);
    assert!(
        (got_f64 - reference_f64).abs() <= ulp_bound_f64(4.0, reference_f64),
        "TanhGrad(1 - 6.7e-9, f64) = {got_f64}, reference {reference_f64}"
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
