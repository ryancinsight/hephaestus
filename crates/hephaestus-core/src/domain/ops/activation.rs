//! Special-function and activation expressions for the unary markers.

use super::UnaryExpr;
use super::{
    EluGradOp, EluOp, ErfOp, ErfcOp, GeluGradOp, GeluOp, GeluTanhGradOp, GeluTanhOp,
    HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, J0Op, J1Op, LgammaOp,
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
        concat!("(1.0 - ", wgsl_erf_expr!(), ")")
    };
}

impl UnaryExpr<Wgsl> for ErfOp {
    const EXPR: &'static str = wgsl_erf_expr!();
}
impl UnaryExpr<CudaC> for ErfOp {
    const EXPR: &'static str = "erf(x)";
}
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

macro_rules! wgsl_gelu_erf_expr {
    () => {
        "(sign((x * 0.7071067811865476)) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) - 1.453152027) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) - 0.284496736) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476)))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs((x * 0.7071067811865476))))) * exp(-(x * 0.7071067811865476) * (x * 0.7071067811865476))))"
    };
}

impl UnaryExpr<Wgsl> for GeluOp {
    const EXPR: &'static str = concat!("(0.5 * x * (1.0 + ", wgsl_gelu_erf_expr!(), "))");
}
impl UnaryExpr<CudaC> for GeluOp {
    const EXPR: &'static str = "0.5f * x * (1.0f + erff(x * 0.7071067811865476f))";
}
impl UnaryExpr<Wgsl> for GeluGradOp {
    const EXPR: &'static str = concat!(
        "(0.5 * (1.0 + ",
        wgsl_gelu_erf_expr!(),
        ") + x * exp(-0.5 * x * x) * 0.3989422804014327)"
    );
}
impl UnaryExpr<CudaC> for GeluGradOp {
    const EXPR: &'static str = "0.5f * (1.0f + erff(x * 0.7071067811865476f)) + x * expf(-0.5f * x * x) * 0.3989422804014327f";
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
    (
        SigmoidOp,
        "1.0 / (1.0 + exp(-x))",
        "1.0f / (1.0f + exp(-x))"
    ),
    (SigmoidGradOp, "x * (1.0 - x)", "x * (1.0f - x)"),
    (TanhOp, "tanh(x)", "tanh(x)"),
    (TanhGradOp, "1.0 - x * x", "1.0f - x * x"),
    (
        GeluTanhOp,
        "0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))",
        "0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
    ),
    (
        GeluTanhGradOp,
        "0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))) + 0.5 * x * (1.0 - tanh(0.7978845608 * (x + 0.044715 * x * x * x)) * tanh(0.7978845608 * (x + 0.044715 * x * x * x))) * 0.7978845608 * (1.0 + 0.134145 * x * x)",
        "0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + 0.5f * x * (1.0f - tanh(0.7978845608f * (x + 0.044715f * x * x * x)) * tanh(0.7978845608f * (x + 0.044715f * x * x * x))) * 0.7978845608f * (1.0f + 0.134145f * x * x)"
    ),
    (SiluOp, "x / (1.0 + exp(-x))", "x / (1.0f + exp(-x))"),
    (
        SiluGradOp,
        "(1.0 / (1.0 + exp(-x))) * (1.0 + x * (1.0 - (1.0 / (1.0 + exp(-x)))))",
        "(1.0f / (1.0f + exp(-x))) * (1.0f + x * (1.0f - (1.0f / (1.0f + exp(-x)))))"
    ),
    (SoftplusOp, "log(1.0 + exp(x))", "log(1.0f + exp(x))"),
    (
        SoftplusGradOp,
        "1.0 / (1.0 + exp(-x))",
        "1.0f / (1.0f + exp(-x))"
    ),
    (
        MishOp,
        "x * tanh(log(1.0 + exp(x)))",
        "x * tanhf(logf(1.0f + expf(x)))"
    ),
    (
        MishGradOp,
        "tanh(log(1.0 + exp(x))) + x * (1.0 - tanh(log(1.0 + exp(x))) * tanh(log(1.0 + exp(x)))) * (1.0 / (1.0 + exp(-x)))",
        "tanhf(logf(1.0f + expf(x))) + x * (1.0f - tanhf(logf(1.0f + expf(x))) * tanhf(logf(1.0f + expf(x)))) * (1.0f / (1.0f + expf(-x)))"
    ),
    (
        EluOp,
        "select(exp(x) - 1.0, x, x >= 0.0)",
        "x >= 0.0f ? x : expf(x) - 1.0f"
    ),
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
