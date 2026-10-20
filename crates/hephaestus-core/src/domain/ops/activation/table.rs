//! Table-driven activation expressions.

use super::super::UnaryExpr;
use super::super::{
    EluGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, ReluOp, SigmoidGradOp, SigmoidOp,
    SoftplusGradOp, SoftsignGradOp, SoftsignOp, TanhGradOp, TanhOp,
};
use crate::domain::dialect::{CudaC, Wgsl};

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
        "x <= 0.0f ? exp(x) / (1.0f + exp(x)) : 1.0f / (1.0f + exp(-x))"
    ),
    (SigmoidGradOp, "x * (1.0 - x)", "x * (1.0f - x)"),
    (TanhOp, "tanh(x)", "tanh(x)"),
    (TanhGradOp, "1.0 - x * x", "1.0f - x * x"),
    // GeluTanhOp, GeluTanhGradOp, SiluOp, and SiluGradOp live in
    // `stable.rs`: their C spellings are macro-built stabilized forms,
    // and the table only holds literals.
    // SoftplusOp stays out of the macro: the WGSL spelling is a
    // machine-built select tree (see below) and the C spelling calls the
    // native `log1p` builtin.
    (
        SoftplusGradOp,
        "1.0 / (1.0 + exp(-x))",
        "x <= 0.0f ? exp(x) / (1.0f + exp(x)) : 1.0f / (1.0f + exp(-x))"
    ),
    // MishOp stays out of the macro: the WGSL spelling wraps the
    // `softplus` fragment (see below) and the C spelling calls the
    // native `log1p` builtin.
    // MishGradOp stays out of the macro: the WGSL spelling guards the
    // `tanh(sp)` lead term by arm (see below) and the C spelling uses
    // the stable `max + log1p` softplus under the `tanh`.
    // EluOp stays out of the macro: the WGSL spelling nests the `expm1`
    // fragment (see below) and the C spelling calls the native builtin.
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
