//! Special-function and activation expressions for the unary markers.

use super::UnaryExpr;
use super::{
    EluGradOp, EluOp, ErfOp, ErfcOp, GeluGradOp, GeluOp, GeluTanhGradOp, GeluTanhOp,
    HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, LgammaOp, MishGradOp, MishOp,
    ReluGradOp, ReluOp, SigmoidGradOp, SigmoidOp, SiluGradOp, SiluOp, SoftplusGradOp, SoftplusOp,
    SoftsignGradOp, SoftsignOp, TanhGradOp, TanhOp,
};
use crate::domain::dialect::{CudaC, Wgsl};

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
        ReluGradOp,
        "select(0.0, 1.0, x > 0.0)",
        "x > 0.0f ? 1.0f : 0.0f"
    ),
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
        HardsigmoidGradOp,
        "select(0.0, 1.0 / 6.0, (x > -3.0) && (x < 3.0))",
        "(x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f"
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
