//! Special-function and activation expressions for the unary markers.

use super::UnaryExpr;
use super::{
    EluGradOp, EluOp, ErfOp, ErfcOp, GeluGradOp, GeluOp, GeluTanhGradOp, GeluTanhOp,
    HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, J0Op, J1Op, K0Op, LgammaOp,
    MishGradOp, MishOp, ReluGradOp, ReluOp, SigmoidGradOp, SigmoidOp, SiluGradOp, SiluOp,
    SoftplusGradOp, SoftplusOp, SoftsignGradOp, SoftsignOp, TanhGradOp, TanhOp,
};
use crate::domain::dialect::{CudaC, HipC, Wgsl};

macro_rules! wgsl_erf_expr {
    () => {
        "(sign((x)) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs((x)))) - 1.453152027) * (1.0 / (1.0 + 0.3275911 * abs((x)))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs((x)))) - 0.284496736) * (1.0 / (1.0 + 0.3275911 * abs((x)))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs((x))))) * exp(-((x)) * ((x)))))"
    };
}

macro_rules! wgsl_erfc_expr {
    () => {
        concat!(
            "select((2.0 - ",
            wgsl_erfc_nr_pos!("(abs(x))"),
            "), ",
            wgsl_erfc_nr_pos!("(abs(x))"),
            ", (x) >= 0.0)"
        )
    };
}

// Shared by `GeluOp` and `GeluGradOp`: `erfc(-(x)·inv√2)` through the same
// two-sided select. The comparison tests `x` directly (`x <= 0` holds
// exactly when the negated argument is non-negative, NaN included), and
// `|x|·inv√2` spells the modulus without negating first.
macro_rules! wgsl_gelu_erfc_expr {
    () => {
        concat!(
            "select((2.0 - ",
            wgsl_erfc_nr_pos!("(abs((x)) * 0.7071067811865476)"),
            "), ",
            wgsl_erfc_nr_pos!("(abs((x)) * 0.7071067811865476)"),
            ", (x) <= 0.0)"
        )
    };
}

impl UnaryExpr<Wgsl> for ErfOp {
    const EXPR: &'static str = wgsl_erf_expr!();
}
impl UnaryExpr<CudaC> for ErfOp {
    const EXPR: &'static str = "erf(x)";
}
// Erfc renders the Numerical Recipes `erfcc` Chebyshev form directly (see
// `expr_macros`): the retired `1 - erf` spelling cancelled to `0` past
// `x = 2.75` while the value stays a normal float down to `x ≈ 9`. The C
// dialects call native `erfc` (4 ULP `erfcf` per the current Programming
// Guide table).
impl UnaryExpr<Wgsl> for ErfcOp {
    const EXPR: &'static str = wgsl_erfc_expr!();
}
impl UnaryExpr<CudaC> for ErfcOp {
    const EXPR: &'static str = "erfc(x)";
}

macro_rules! wgsl_lgamma_positive_expr {
    ($value:literal) => {
        concat!(
            "(0.9189385332046727 + ((",
            $value,
            ") - 0.5) * log((",
            $value,
            ") + 6.5) - ((",
            $value,
            ") + 6.5) + log(0.9999999999998099 + 676.5203681218851 / (",
            $value,
            ") - 1259.1392167224028 / ((",
            $value,
            ") + 1.0) + 771.3234287776531 / ((",
            $value,
            ") + 2.0) - 176.6150291621406 / ((",
            $value,
            ") + 3.0) + 12.5073432786869 / ((",
            $value,
            ") + 4.0) - 0.13857109526572 / ((",
            $value,
            ") + 5.0) + 9.98436957801957e-6 / ((",
            $value,
            ") + 6.0) + 1.50563273514931e-7 / ((",
            $value,
            ") + 7.0)))"
        )
    };
}

macro_rules! wgsl_lgamma_expr {
    () => {
        concat!(
            "select(select(select(",
            "(1.1447298858494002 - log(abs(sin(3.141592653589793 * x))) - ",
            wgsl_lgamma_positive_expr!("max(abs(1.0 - x), 0.5)"),
            "), ",
            wgsl_lgamma_positive_expr!("max(abs(x), 0.5)"),
            ", x >= 0.5), 1.0 / abs(x - trunc(x)), ((x <= 0.0) && (x == trunc(x)))), abs(x), abs(x) > 3.402823466e+38)"
        )
    };
}

impl UnaryExpr<Wgsl> for LgammaOp {
    const EXPR: &'static str = wgsl_lgamma_expr!();
}
impl UnaryExpr<CudaC> for LgammaOp {
    const EXPR: &'static str = "lgamma(x)";
}

// Gelu renders `0.5·x·erfc(-x/√2)` (the `1 + erf(z) = erfc(-z)` identity,
// ADR 0061 Decision 6): the textbook `1 + erf` rounds to `0` for
// `x <= -5.5` while the value stays a normal float. The C dialects call
// native `erfcf` (4 ULP per the current Programming Guide table); the
// float-suffixed builtins and constant keep their existing spelling, so the
// change is the identity alone. `x = -∞` guards to `-0` (the unguarded
// form makes `-∞·0`); the comparison doubles as an f64 lane guard, where
// magnitudes past `FLT_MAX` share the `-0` limit.
impl UnaryExpr<Wgsl> for GeluOp {
    const EXPR: &'static str = concat!(
        "select((0.5 * (x) * ",
        wgsl_gelu_erfc_expr!(),
        "), -0.0, (x) < -3.402823466e+38)"
    );
}
impl UnaryExpr<CudaC> for GeluOp {
    const EXPR: &'static str =
        "(x) < -3.402823466e+38f ? -0.0f : 0.5f * x * erfcf(-(x * 0.7071067811865476f))";
}
// The gradient renders `0.5·erfc(-x/√2)` plus the untouched Gaussian term
// (its decay is the true asymptotic, not a cancellation artifact), taking
// the input per ADR 0061 Decision 7. Both infinities guard explicitly: the
// Gaussian term makes `±∞·0` unguarded, while the analytic limits are `1`
// at `+∞` and `0` at `-∞`.
impl UnaryExpr<Wgsl> for GeluGradOp {
    const EXPR: &'static str = concat!(
        "select(select((0.5 * ",
        wgsl_gelu_erfc_expr!(),
        " + (x) * exp(-0.5 * (x) * (x)) * 0.3989422804014327), 1.0, (x) > 3.402823466e+38), 0.0, (x) < -3.402823466e+38)"
    );
}
impl UnaryExpr<CudaC> for GeluGradOp {
    const EXPR: &'static str = "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : 0.5f * erfcf(-(x * 0.7071067811865476f)) + x * expf(-0.5f * x * x) * 0.3989422804014327f)";
}

macro_rules! impl_activation_unary_exprs {
    ($(($op:ty, $wgsl:literal, $cuda:literal)),+ $(,)?) => {
        $(
            impl UnaryExpr<Wgsl> for $op {
                const EXPR: &'static str = $wgsl;
            }
            impl UnaryExpr<CudaC> for $op {
                const EXPR: &'static str = $cuda;
            }
        )+
    };
}

impl_activation_unary_exprs!(
    (ReluOp, "max(x, 0.0)", "max(x, 0.0f)"),
    (SigmoidGradOp, "x * (1.0 - x)", "x * (1.0f - x)"),
    (TanhOp, "tanh(x)", "tanh(x)"),
    (TanhGradOp, "1.0 - x * x", "1.0f - x * x"),
    (
        EluGradOp,
        "select(exp(x), 1.0, x >= 0.0)",
        "x >= 0.0f ? 1.0f : expf(x)"
    ),
    (
        HardsigmoidOp,
        "clamp(x / 6.0 + 0.5, 0.0, 1.0)",
        "fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)"
    ),
    (
        HardswishOp,
        "x * clamp(x + 3.0, 0.0, 6.0) / 6.0",
        "x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f"
    ),
    (
        HardswishGradOp,
        "select(select(0.0, (2.0 * x + 3.0) / 6.0, (x > -3.0) && (x < 3.0)), 1.0, x >= 3.0)",
        "x >= 3.0f ? 1.0f : (x > -3.0f ? (2.0f * x + 3.0f) / 6.0f : 0.0f)"
    ),
    (SoftsignOp, "x / (1.0 + abs(x))", "x / (1.0f + fabsf(x))"),
    (
        SoftsignGradOp,
        "1.0 / ((1.0 + abs(x)) * (1.0 + abs(x)))",
        "1.0f / ((1.0f + fabsf(x)) * (1.0f + fabsf(x)))"
    ),
);

// Sigmoid, softplus, and the softplus gradient stay out of the macro: each
// renders the two-sided stabilized form, and the WGSL softplus splices the
// shared `log1p` core, which the literal-only table rows cannot spell. The
// one-sided spellings overflow `exp` past `|x| = ln(FLT_MAX)`: softplus
// returned `inf` for finite values above `x = 88.73` and `0` for normal
// floats below `x = -16.6` (`1 + exp(x)` rounding to `1`), while the
// sigmoid quotient collapsed to `0` below `x = -88.73`. The C dialects
// branch on the sign and call native `log1p`/`exp` (unsuffixed so overload
// resolution picks the lane width, as `SinOp` does); the maxima shift is
// exact by Sterbenz. The WGSL sigmoid branches the same way over `select`
// (expression arms, so `SUPPORTS_F64` keeps its default). The WGSL softplus
// evaluates `log1p(exp(-|x|))` through two bands of the shared core — the
// core band plus the exact upper shift — since `exp` of a non-positive
// argument never leaves `(0, 1]` and the outer `log` bands of
// `wgsl_log1p_expr` are unreachable there; the band edge and shift below
// repeat that macro's definitions. The form is `log`-free, so the canary
// `log` gate no longer refuses its `f64` dispatch (the gate test now pins
// `LnOp` instead); every special input takes an exact arm (`±0` through
// the negative half, infinities through the shift, NaN through `x`).
macro_rules! wgsl_softplus_expr {
    () => {
        concat!(
            "select((x) + ",
            "select(",
            wgsl_l1p_core!("(exp(-(x)))"),
            ", (0.6931471805599453 + ",
            wgsl_l1p_core!("(((exp(-(x))) - 1.0) * 0.5)"),
            "), (exp(-(x))) > 0.4142135623730951)",
            ", ",
            "select(",
            wgsl_l1p_core!("(exp((x)))"),
            ", (0.6931471805599453 + ",
            wgsl_l1p_core!("(((exp((x))) - 1.0) * 0.5)"),
            "), (exp((x))) > 0.4142135623730951)",
            ", (x) <= 0.0)"
        )
    };
}

impl UnaryExpr<Wgsl> for SigmoidOp {
    const EXPR: &'static str =
        "select((1.0 / (1.0 + exp(-x))), (exp(x) / (1.0 + exp(x))), x <= 0.0)";
}
impl UnaryExpr<CudaC> for SigmoidOp {
    const EXPR: &'static str = "x <= 0.0f ? exp(x) / (1.0f + exp(x)) : 1.0f / (1.0f + exp(-x))";
}
impl UnaryExpr<Wgsl> for SoftplusGradOp {
    const EXPR: &'static str =
        "select((1.0 / (1.0 + exp(-x))), (exp(x) / (1.0 + exp(x))), x <= 0.0)";
}
impl UnaryExpr<CudaC> for SoftplusGradOp {
    const EXPR: &'static str = "x <= 0.0f ? exp(x) / (1.0f + exp(x)) : 1.0f / (1.0f + exp(-x))";
}
impl UnaryExpr<Wgsl> for SoftplusOp {
    const EXPR: &'static str = wgsl_softplus_expr!();
}
impl UnaryExpr<CudaC> for SoftplusOp {
    const EXPR: &'static str = "x <= 0.0f ? log1p(exp(x)) : x + log1p(exp(-x))";
}

// Elu stays out of the macro: its negative branch renders the `Expm1Op`
// form (Taylor core under `|x| <= 0.3`, direct `exp(x) - 1` outside),
// while the C dialect calls native `expm1` (1 ULP per the CUDA Table 17 /
// HIP math API rows; unsuffixed so overload resolution picks the lane
// width). The positive branch and the kink select are exact.
impl UnaryExpr<Wgsl> for EluOp {
    const EXPR: &'static str = "select(select((exp(x) - 1.0), (x) * (1.0 + (x) * (0.5 + (x) * (0.16666666666666666 + (x) * (0.041666666666666664 + (x) * (0.008333333333333333 + (x) * 0.001388888888888889))))), abs(x) <= 0.3), x, x >= 0.0)";
}
impl UnaryExpr<CudaC> for EluOp {
    const EXPR: &'static str = "x >= 0.0f ? x : expm1(x)";
}

// The self-gated family (SiLU, Mish, tanh-GELU, forward and gradient)
// composes three shared kernels, mirroring `activation_value::stable`:
// the branch-selected sigmoid, the two-sided softplus, and the gelu-tanh
// inner argument doubled. Each composer parents its output, so assemblies
// splice it under `*`, `/`, comparisons, and ternary conditions alike.
// Macro arguments are fully parenthesized literals, as the Bessel macros
// require. The C assemblies serve both C dialects (as `c_j0_expr` does):
// CUDA verifies them on-device, and the pins below record the shared
// spelling until HIP hardware says otherwise.
macro_rules! wgsl_stable_sigmoid {
    ($w:literal) => {
        concat!(
            "(select((1.0 / (1.0 + exp(-(",
            $w,
            ")))), ((exp((",
            $w,
            "))) / (1.0 + (exp((",
            $w,
            "))))), (",
            $w,
            ") <= 0.0))"
        )
    };
}

macro_rules! c_stable_sigmoid {
    ($w:literal) => {
        concat!(
            "((",
            $w,
            ") <= 0.0f ? exp((",
            $w,
            ")) / (1.0f + exp((",
            $w,
            "))) : 1.0f / (1.0f + exp(-(",
            $w,
            "))))"
        )
    };
}

// The `SoftplusOp` C spelling, parenthesized so the mish assemblies splice
// it under `tanh`, `*`, and the `sp` crossover comparison.
macro_rules! c_softplus {
    () => {
        "((x) <= 0.0f ? log1p(exp((x))) : (x) + log1p(exp(-((x)))))"
    };
}

// `w = 2z`, `z = c0·(x + c1·x³)`, spelled twice from one source: both halves
// evaluate identically, and doubling is exact short of overflow, matching
// the value function (`z + z`) bit for bit including the overflow to `±∞`.
macro_rules! wgsl_gelu_tanh_w {
    () => {
        "((0.7978845608 * (x + 0.044715 * x * x * x)) + (0.7978845608 * (x + 0.044715 * x * x * x)))"
    };
}

macro_rules! c_gelu_tanh_w {
    () => {
        "((0.7978845608f * (x + 0.044715f * x * x * x)) + (0.7978845608f * (x + 0.044715f * x * x * x)))"
    };
}

// Fixed-`w` twins of the sigmoid composers: `macro_rules!` `:literal`
// cannot carry a macro expansion, so `w` splices through `concat!`.
macro_rules! wgsl_gelu_tanh_s {
    () => {
        concat!(
            "(select((1.0 / (1.0 + exp(-(",
            wgsl_gelu_tanh_w!(),
            ")))), ((exp((",
            wgsl_gelu_tanh_w!(),
            "))) / (1.0 + (exp((",
            wgsl_gelu_tanh_w!(),
            "))))), (",
            wgsl_gelu_tanh_w!(),
            ") <= 0.0))"
        )
    };
}

macro_rules! c_gelu_tanh_s {
    () => {
        concat!(
            "((",
            c_gelu_tanh_w!(),
            ") <= 0.0f ? exp((",
            c_gelu_tanh_w!(),
            ")) / (1.0f + exp((",
            c_gelu_tanh_w!(),
            "))) : 1.0f / (1.0f + exp(-(",
            c_gelu_tanh_w!(),
            "))))"
        )
    };
}

// The independent saturation `s·σ(−w)`, pre-grouped like the value
// function. `σ(−w)` spells its negative half directly: this composer only
// splices past the `w > 2.47` crossover, where `−w < 0` always, so the
// positive half is unreachable and the direct half is exactly what the
// branch-selected sigmoid computes there (NaN included: both spellings
// propagate it through a failed comparison).
macro_rules! wgsl_gelu_tanh_sat_indep {
    () => {
        concat!(
            "(",
            wgsl_gelu_tanh_s!(),
            " * (exp(-(",
            wgsl_gelu_tanh_w!(),
            ")) / (1.0 + exp(-(",
            wgsl_gelu_tanh_w!(),
            ")))))"
        )
    };
}

macro_rules! c_gelu_tanh_sat_indep {
    () => {
        concat!(
            "(",
            c_gelu_tanh_s!(),
            " * (exp(-(",
            c_gelu_tanh_w!(),
            ")) / (1.0f + exp(-(",
            c_gelu_tanh_w!(),
            ")))))"
        )
    };
}

// `sech²(sp) = 4u/(1+u)²`, `u = exp(−2·sp)`, for the mish-grad crossover.
macro_rules! wgsl_mish_sech2 {
    () => {
        concat!(
            "((4.0 * exp(-2.0 * ",
            wgsl_softplus_expr!(),
            ")) / ((1.0 + exp(-2.0 * ",
            wgsl_softplus_expr!(),
            ")) * (1.0 + exp(-2.0 * ",
            wgsl_softplus_expr!(),
            "))))"
        )
    };
}

macro_rules! c_mish_sech2 {
    () => {
        concat!(
            "((4.0f * exp(-2.0f * ",
            c_softplus!(),
            ")) / ((1.0f + exp(-2.0f * ",
            c_softplus!(),
            ")) * (1.0f + exp(-2.0f * ",
            c_softplus!(),
            "))))"
        )
    };
}

macro_rules! wgsl_silu_expr {
    () => {
        concat!(
            "select((x) * ",
            wgsl_stable_sigmoid!("(x)"),
            ", -0.0, (x) < -3.402823466e+38)"
        )
    };
}

macro_rules! c_silu_expr {
    () => {
        concat!(
            "(x) < -3.402823466e+38f ? -0.0f : (x) * ",
            c_stable_sigmoid!("(x)")
        )
    };
}

macro_rules! wgsl_silu_grad_expr {
    () => {
        concat!(
            "select(select(",
            wgsl_stable_sigmoid!("(x)"),
            " * (1.0 + (x) * select(",
            wgsl_stable_sigmoid!("(-(x))"),
            ", (1.0 - ",
            wgsl_stable_sigmoid!("(x)"),
            "), (x) <= 2.63)), 1.0, (x) > 3.402823466e+38), 0.0, (x) < -3.402823466e+38)"
        )
    };
}

macro_rules! c_silu_grad_expr {
    () => {
        concat!(
            "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : ",
            c_stable_sigmoid!("(x)"),
            " * (1.0f + (x) * ((x) <= 2.63f ? 1.0f - ",
            c_stable_sigmoid!("(x)"),
            " : ",
            c_stable_sigmoid!("(-(x))"),
            ")))"
        )
    };
}

macro_rules! wgsl_mish_expr {
    () => {
        concat!(
            "select((x) * tanh(",
            wgsl_softplus_expr!(),
            "), -0.0, (x) < -3.402823466e+38)"
        )
    };
}

macro_rules! c_mish_expr {
    () => {
        concat!(
            "(x) < -3.402823466e+38f ? -0.0f : (x) * tanh(",
            c_softplus!(),
            ")"
        )
    };
}

macro_rules! wgsl_mish_grad_expr {
    () => {
        concat!(
            "select(select(tanh(",
            wgsl_softplus_expr!(),
            ") + (x) * select(",
            wgsl_mish_sech2!(),
            ", (1.0 - tanh(",
            wgsl_softplus_expr!(),
            ") * tanh(",
            wgsl_softplus_expr!(),
            ")), (",
            wgsl_softplus_expr!(),
            ") <= 1.55) * ",
            wgsl_stable_sigmoid!("(x)"),
            ", 1.0, (x) > 3.402823466e+38), 0.0, (x) < -3.402823466e+38)"
        )
    };
}

macro_rules! c_mish_grad_expr {
    () => {
        concat!(
            "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : tanh(",
            c_softplus!(),
            ") + (x) * ((",
            c_softplus!(),
            ") <= 1.55f ? 1.0f - tanh(",
            c_softplus!(),
            ") * tanh(",
            c_softplus!(),
            ") : ",
            c_mish_sech2!(),
            ") * ",
            c_stable_sigmoid!("(x)"),
            ")"
        )
    };
}

macro_rules! wgsl_gelu_tanh_expr {
    () => {
        concat!(
            "select((x) * ",
            wgsl_gelu_tanh_s!(),
            ", -0.0, (x) < -3.402823466e+38)"
        )
    };
}

macro_rules! c_gelu_tanh_expr {
    () => {
        concat!("(x) < -3.402823466e+38f ? -0.0f : (x) * ", c_gelu_tanh_s!())
    };
}

macro_rules! wgsl_gelu_tanh_grad_expr {
    () => {
        concat!(
            "select(select(select(select(",
            wgsl_gelu_tanh_s!(),
            " + 2.0 * (x) * ",
            wgsl_gelu_tanh_sat_indep!(),
            " * 0.7978845608 * (1.0 + 3.0 * 0.044715 * (x) * (x)), ",
            wgsl_gelu_tanh_s!(),
            ", ",
            wgsl_gelu_tanh_sat_indep!(),
            " == 0.0), select(",
            wgsl_gelu_tanh_s!(),
            " + 2.0 * (x) * ",
            wgsl_gelu_tanh_s!(),
            " * (1.0 - ",
            wgsl_gelu_tanh_s!(),
            ") * 0.7978845608 * (1.0 + 3.0 * 0.044715 * (x) * (x)), ",
            wgsl_gelu_tanh_s!(),
            ", ",
            wgsl_gelu_tanh_s!(),
            " * (1.0 - ",
            wgsl_gelu_tanh_s!(),
            ") == 0.0), (",
            wgsl_gelu_tanh_w!(),
            ") <= 2.47), 1.0, (x) > 3.402823466e+38), 0.0, (x) < -3.402823466e+38)"
        )
    };
}

macro_rules! c_gelu_tanh_grad_expr {
    () => {
        concat!(
            "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : ((",
            c_gelu_tanh_w!(),
            ") <= 2.47f ? (",
            c_gelu_tanh_s!(),
            " * (1.0f - ",
            c_gelu_tanh_s!(),
            ") == 0.0f ? ",
            c_gelu_tanh_s!(),
            " : ",
            c_gelu_tanh_s!(),
            " + 2.0f * (x) * ",
            c_gelu_tanh_s!(),
            " * (1.0f - ",
            c_gelu_tanh_s!(),
            ") * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x))) : (",
            c_gelu_tanh_sat_indep!(),
            " == 0.0f ? ",
            c_gelu_tanh_s!(),
            " : ",
            c_gelu_tanh_s!(),
            " + 2.0f * (x) * ",
            c_gelu_tanh_sat_indep!(),
            " * 0.7978845608f * (1.0f + 3.0f * 0.044715f * (x) * (x)))))"
        )
    };
}

// `x·sigmoid(x)` through the branch-selected sigmoid (ADR 0061 Decision 6):
// the retired `x / (1 + exp(-x))` overflowed `exp` past `x = -88.73` and made
// `-∞/∞` at `x = -∞`. The guard catches only `-∞` (finite inputs stay past
// `-FLT_MAX`); `+∞` needs none (`∞·1`), and NaN propagates through the
// failed comparisons.
impl UnaryExpr<Wgsl> for SiluOp {
    const EXPR: &'static str = wgsl_silu_expr!();
}
impl UnaryExpr<CudaC> for SiluOp {
    const EXPR: &'static str = c_silu_expr!();
}
impl UnaryExpr<HipC> for SiluOp {
    const EXPR: &'static str = c_silu_expr!();
}

// `s·(1 + x·(1−s))`, taking the input per ADR 0061 Decision 7. `1 − s` goes
// direct below `x = 2.63` and independent above (`SILU_GRAD_CROSSOVER`);
// the `f` suffix matches `T::from_f64` in the f32 lane, and in the f64 lane
// it converts (like every literal here), moving the switch by half an f32
// ULP inside the measured alternation band, where both forms hold ~2 ULP.
// Both infinities guard explicitly (`±∞·0` unguarded); NaN propagates.
impl UnaryExpr<Wgsl> for SiluGradOp {
    const EXPR: &'static str = wgsl_silu_grad_expr!();
}
impl UnaryExpr<CudaC> for SiluGradOp {
    const EXPR: &'static str = c_silu_grad_expr!();
}
impl UnaryExpr<HipC> for SiluGradOp {
    const EXPR: &'static str = c_silu_grad_expr!();
}

// `x·tanh(softplus(x))`, composing the two-sided softplus (ADR 0061
// Decision 6): the retired `log(1 + exp(x))` overflowed `exp` past
// `x = 88.73` and rounded `1 + exp(x)` to `1` below `x = -16.6`, zeroing the
// whole negative tail. Same `-∞` guard as SiLU; `tanh` keeps the canary's
// f64 refusal on tanh-crash adapters, as `TanhOp` itself does.
impl UnaryExpr<Wgsl> for MishOp {
    const EXPR: &'static str = wgsl_mish_expr!();
}
impl UnaryExpr<CudaC> for MishOp {
    const EXPR: &'static str = c_mish_expr!();
}
impl UnaryExpr<HipC> for MishOp {
    const EXPR: &'static str = c_mish_expr!();
}

// `t + x·(1−t²)·σ(x)` with `t = tanh(sp)`, taking the input per ADR 0061
// Decision 7. `1 − t²` goes direct below `sp = 1.55` and `sech²` above
// (`MISH_GRAD_CROSSOVER`, compared on `sp`, not `t`); both infinities guard
// explicitly. The softplus splices seven times (value, crossover, and the
// three `sech²` exponentials); WGSL evaluates both `select` arms, so the
// dead arm's overflow is discarded, never trapped.
impl UnaryExpr<Wgsl> for MishGradOp {
    const EXPR: &'static str = wgsl_mish_grad_expr!();
}
impl UnaryExpr<CudaC> for MishGradOp {
    const EXPR: &'static str = c_mish_grad_expr!();
}
impl UnaryExpr<HipC> for MishGradOp {
    const EXPR: &'static str = c_mish_grad_expr!();
}

// `x·σ(2z)` via `0.5·(1 + tanh(z)) = σ(2z)` (ADR 0061 Decision 6): the
// retired `1 + tanh(z)` cancelled past `z = -4.4`, and the new form calls no
// `tanh` at all, so f64 dispatches where the retired spelling refused on
// tanh-crash adapters. Same `-∞` guard as SiLU.
impl UnaryExpr<Wgsl> for GeluTanhOp {
    const EXPR: &'static str = wgsl_gelu_tanh_expr!();
}
impl UnaryExpr<CudaC> for GeluTanhOp {
    const EXPR: &'static str = c_gelu_tanh_expr!();
}
impl UnaryExpr<HipC> for GeluTanhOp {
    const EXPR: &'static str = c_gelu_tanh_expr!();
}

// `s + 2x·s(1−s)·c0·(1 + 3c1·x²)` in sigmoid form, taking the input per ADR
// 0061 Decision 7. `1 − s` goes direct below `w = 2.47` and independent
// above (`GELU_TANH_GRAD_CROSSOVER`); each branch keeps the value function's
// measured multiplication order (direct multiplies `s` and `1 − s`
// separately, independent pre-groups the saturation). Either saturation at
// exactly zero returns `s`: past `|x| ≈ 5e19` the `x²` factor overflows, and
// forming the product would make `0·∞` (the value function's early-out,
// mirrored here; the C ternary short-circuits, the WGSL `select` discards
// the dead arm). Both infinities guard explicitly.
impl UnaryExpr<Wgsl> for GeluTanhGradOp {
    const EXPR: &'static str = wgsl_gelu_tanh_grad_expr!();
}
impl UnaryExpr<CudaC> for GeluTanhGradOp {
    const EXPR: &'static str = c_gelu_tanh_grad_expr!();
}
impl UnaryExpr<HipC> for GeluTanhGradOp {
    const EXPR: &'static str = c_gelu_tanh_grad_expr!();
}

// The two gradient masks stay out of the macro: their WGSL spellings are
// `select` with literal-only arms, which concretize to `f32` and reject
// `f64` output buffers, so they report `SUPPORTS_F64 = false` until a
// typed unary seam lands. The CUDA spellings convert implicitly and serve
// both precisions.
impl UnaryExpr<Wgsl> for ReluGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, x > 0.0)";
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for ReluGradOp {
    const EXPR: &'static str = "x > 0.0f ? 1.0f : 0.0f";
}
impl UnaryExpr<Wgsl> for HardsigmoidGradOp {
    const EXPR: &'static str = "select(0.0, 1.0 / 6.0, (x > -3.0) && (x < 3.0))";
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for HardsigmoidGradOp {
    const EXPR: &'static str = "(x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f";
}

// Bessel J0/J1 share one structure across dialects: the Numerical-Recipes
// rational approximation for `|x| < 8` and Hankel's asymptotic expansion
// otherwise, with the same coefficients and nesting as leto's scalar
// `j0`/`j1`. Only the absolute-value spelling differs (`abs` vs `fabs`),
// so the branch macros take it as `$ax`; `sin`/`cos`/`sqrt` stay unsuffixed
// so C overload resolution picks the lane width, as `SinOp` does.
macro_rules! bessel_y2 {
    () => {
        "((x) * (x))"
    };
}

macro_rules! bessel_z {
    ($ax:literal) => {
        concat!("(8.0 / (", $ax, "))")
    };
}

macro_rules! bessel_yh {
    ($ax:literal) => {
        concat!("((8.0 / (", $ax, ")) * (8.0 / (", $ax, ")))")
    };
}

macro_rules! bessel_xx {
    ($ax:literal, $off:literal) => {
        concat!("((", $ax, ") - ", $off, ")")
    };
}

macro_rules! j0_rational {
    () => {
        concat!(
            "((",
            "57568490574.0 + ",
            bessel_y2!(),
            "*(-13362590354.0 + ",
            bessel_y2!(),
            "*(651619640.7 + ",
            bessel_y2!(),
            "*(-11214424.18 + ",
            bessel_y2!(),
            "*(77392.33017 + ",
            bessel_y2!(),
            "*-184.9052456))))",
            ") / (",
            "57568490411.0 + ",
            bessel_y2!(),
            "*(1029532985.0 + ",
            bessel_y2!(),
            "*(9494680.718 + ",
            bessel_y2!(),
            "*(59272.64853 + ",
            bessel_y2!(),
            "*(267.8532712 + ",
            bessel_y2!(),
            "))))",
            "))"
        )
    };
}

macro_rules! j0_hankel {
    ($ax:literal) => {
        concat!(
            "(sqrt(2.0 / (3.141592653589793 * (",
            $ax,
            "))) * ((",
            "1.0 + ",
            bessel_yh!($ax),
            "*(-0.001098628627 + ",
            bessel_yh!($ax),
            "*(0.000002734510407 + ",
            bessel_yh!($ax),
            "*(-2.073370639e-6 + ",
            bessel_yh!($ax),
            "*2.093887211e-7)))",
            ") * cos(",
            bessel_xx!($ax, "0.7853981633974483"),
            ") - (",
            bessel_z!($ax),
            ") * (",
            "-0.01562499995 + ",
            bessel_yh!($ax),
            "*(0.0001430488765 + ",
            bessel_yh!($ax),
            "*(-6.911147651e-5 + ",
            bessel_yh!($ax),
            "*(7.621095161e-5 - ",
            bessel_yh!($ax),
            "*9.34935152e-7)))",
            ") * sin(",
            bessel_xx!($ax, "0.7853981633974483"),
            ")))"
        )
    };
}

macro_rules! j1_rational {
    () => {
        concat!(
            "(((x) * (72362614232.0 + ",
            bessel_y2!(),
            "*(-7895059235.0 + ",
            bessel_y2!(),
            "*(242396853.1 + ",
            bessel_y2!(),
            "*(-2972611.439 + ",
            bessel_y2!(),
            "*(15704.48260 + ",
            bessel_y2!(),
            "*-30.16036606))))))) / (",
            "144725228442.0 + ",
            bessel_y2!(),
            "*(2300535178.0 + ",
            bessel_y2!(),
            "*(18583304.74 + ",
            bessel_y2!(),
            "*(99447.43394 + ",
            bessel_y2!(),
            "*(376.9991397 + ",
            bessel_y2!(),
            ")))))"
        )
    };
}

macro_rules! j1_hankel {
    ($ax:literal) => {
        concat!(
            "(sqrt(2.0 / (3.141592653589793 * (",
            $ax,
            "))) * ((",
            "1.0 + ",
            bessel_yh!($ax),
            "*(0.183105e-2 + ",
            bessel_yh!($ax),
            "*(-3.516396496e-5 + ",
            bessel_yh!($ax),
            "*(2.457520174e-5 - ",
            bessel_yh!($ax),
            "*2.400505341e-7)))",
            ") * cos(",
            bessel_xx!($ax, "2.356194490192345"),
            ") - (",
            bessel_z!($ax),
            ") * (",
            "0.04687499995 + ",
            bessel_yh!($ax),
            "*(-0.2002690873e-3 + ",
            bessel_yh!($ax),
            "*(8.449199096e-5 + ",
            bessel_yh!($ax),
            "*(-8.8228987e-5 + ",
            bessel_yh!($ax),
            "*1.050343160e-6)))",
            ") * sin(",
            bessel_xx!($ax, "2.356194490192345"),
            ")))"
        )
    };
}

macro_rules! wgsl_j0_expr {
    () => {
        concat!(
            "select(select(",
            j0_hankel!("abs(x)"),
            ", ",
            j0_rational!(),
            ", abs(x) < 8.0), 1.0, x == 0.0)"
        )
    };
}

macro_rules! wgsl_j1_expr {
    () => {
        concat!(
            "select((",
            j1_hankel!("abs(x)"),
            ") * sign(x), ",
            j1_rational!(),
            ", abs(x) < 8.0)"
        )
    };
}

macro_rules! c_j0_expr {
    () => {
        concat!(
            "((x) == 0.0 ? 1.0 : ((fabs(x) < 8.0) ? (",
            j0_rational!(),
            ") : (",
            j0_hankel!("fabs(x)"),
            ")))"
        )
    };
}

macro_rules! c_j1_expr {
    () => {
        concat!(
            "((fabs(x) < 8.0) ? (",
            j1_rational!(),
            ") : ((",
            j1_hankel!("fabs(x)"),
            ") * ((x) < 0.0 ? -1.0 : 1.0)))"
        )
    };
}

// J0/J1 stay out of the tables: the WGSL spellings nest `select`, which
// concretizes to `f32` exactly as `SignOp` documents, so they report
// `SUPPORTS_F64 = false` until an f64-capable device proves the spelling
// (the arms are x-expressions, so re-enabling is a flag flip once proven).
// The C spellings use unsuffixed double coefficients with `fabs`/`sin`/
// `cos`/`sqrt` overloads, converting implicitly and serving both precisions.
// CUDA and HIP share one spelling through `c_j0_expr!`/`c_j1_expr!`.
impl UnaryExpr<Wgsl> for J0Op {
    const EXPR: &'static str = wgsl_j0_expr!();
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for J0Op {
    const EXPR: &'static str = c_j0_expr!();
}
impl UnaryExpr<HipC> for J0Op {
    const EXPR: &'static str = c_j0_expr!();
}
impl UnaryExpr<Wgsl> for J1Op {
    const EXPR: &'static str = wgsl_j1_expr!();
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for J1Op {
    const EXPR: &'static str = c_j1_expr!();
}
impl UnaryExpr<HipC> for J1Op {
    const EXPR: &'static str = c_j1_expr!();
}

// K0 follows Abramowitz & Stegun 9.8.5 (`0 < x <= 2`: I0 polynomial plus a
// log correction) and 9.8.6 (`x > 2`: scaled exponential series), with the
// same coefficients and nesting as leto's scalar `bessel_k0`. The domain
// guard (`x > 0` and finite, else NaN) needs a NaN literal, which no dialect
// spells: both families use `(x - x) / (x - x)`, NaN for every lane value
// (0/0 and NaN/NaN alike) yet never constant (the lane value is runtime).
// WGSL tests finiteness as `abs(x) <= 3.402823466e+38`, the lgamma-spelling
// magnitude guard: self-subtraction folds to zero under driver fast-math,
// while the magnitude compare is unfoldable and rejects NaN through the
// false comparison. C uses the `isfinite` builtin instead.
macro_rules! k0_t1 {
    () => {
        "(((x) / 3.75) * ((x) / 3.75))"
    };
}

macro_rules! k0_t2 {
    () => {
        "(((x) * 0.5) * ((x) * 0.5))"
    };
}

macro_rules! k0_i0 {
    () => {
        concat!(
            "1.0 + ",
            k0_t1!(),
            "*(3.5156229 + ",
            k0_t1!(),
            "*(3.0899424 + ",
            k0_t1!(),
            "*(1.2067492 + ",
            k0_t1!(),
            "*(0.2659732 + ",
            k0_t1!(),
            "*(0.0360768 + ",
            k0_t1!(),
            "*0.0045813)))))"
        )
    };
}

macro_rules! k0_correction {
    () => {
        concat!(
            "-0.57721566 + ",
            k0_t2!(),
            "*(0.42278420 + ",
            k0_t2!(),
            "*(0.23069756 + ",
            k0_t2!(),
            "*(0.03488590 + ",
            k0_t2!(),
            "*(0.00262698 + ",
            k0_t2!(),
            "*(0.00010750 + ",
            k0_t2!(),
            "*7.4e-6)))))",
        )
    };
}

macro_rules! k0_small {
    () => {
        concat!(
            "(log(((x) * 0.5)) * (-(",
            k0_i0!(),
            ")) + (",
            k0_correction!(),
            "))",
        )
    };
}

macro_rules! k0_t {
    () => {
        "(2.0 / (x))"
    };
}

macro_rules! k0_series {
    () => {
        concat!(
            "1.25331414 + ",
            k0_t!(),
            "*(-0.07832358 + ",
            k0_t!(),
            "*(0.02189568 + ",
            k0_t!(),
            "*(-0.01062446 + ",
            k0_t!(),
            "*(0.00587872 + ",
            k0_t!(),
            "*(-0.00251540 + ",
            k0_t!(),
            "*0.00053208)))))",
        )
    };
}

macro_rules! k0_large {
    () => {
        concat!("((exp(-(x)) / sqrt(x)) * (", k0_series!(), "))",)
    };
}

macro_rules! k0_nan {
    () => {
        "(((x) - (x)) / ((x) - (x)))"
    };
}

macro_rules! wgsl_k0_expr {
    () => {
        concat!(
            "select(",
            k0_nan!(),
            ", select(",
            k0_large!(),
            ", ",
            k0_small!(),
            ", (x) <= 2.0), ((x) > 0.0) && (abs(x) <= 3.402823466e+38))",
        )
    };
}

macro_rules! c_k0_expr {
    () => {
        concat!(
            "(((x) > 0.0 && isfinite(x)) ? (((x) <= 2.0) ? (",
            k0_small!(),
            ") : (",
            k0_large!(),
            ")) : (",
            k0_nan!(),
            "))",
        )
    };
}

// K0 stays out of the tables with J0/J1: the WGSL spelling nests `select`
// and reports `SUPPORTS_F64 = false` per the select precedent, while the C
// spelling converts implicitly and serves both precisions. CUDA and HIP
// share one spelling through `c_k0_expr!`.
impl UnaryExpr<Wgsl> for K0Op {
    const EXPR: &'static str = wgsl_k0_expr!();
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for K0Op {
    const EXPR: &'static str = c_k0_expr!();
}
impl UnaryExpr<HipC> for K0Op {
    const EXPR: &'static str = c_k0_expr!();
}
