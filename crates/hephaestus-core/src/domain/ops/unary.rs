//! Unary operation markers and their math-function expressions.

use super::UnaryExpr;
use crate::domain::dialect::{CudaC, HipC, Wgsl};

// ── Unary markers ────────────────────────────────────────────────────────

/// Exponential operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpOp;

/// Fused negated exponential `exp(-x)` marker (e.g. Beer–Lambert
/// transmission).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpNegOp;

/// Natural logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LnOp;

/// Sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinOp;

/// Cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosOp;

/// Square-root operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SqrtOp;

/// Absolute value operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AbsOp;

/// Negation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct NegOp;

/// Reciprocal operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecipOp;

/// Identity/copy operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityOp;

/// Tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanOp;

/// Arcsine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AsinOp;

/// Arccosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcosOp;

/// Arctangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtanOp;

/// Hyperbolic sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinhOp;

/// Hyperbolic cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoshOp;

/// Base-two logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log2Op;

/// Base-ten logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log10Op;

/// Base-two exponential operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exp2Op;

/// Inverse hyperbolic tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtanhOp;

/// Inverse hyperbolic sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AsinhOp;

/// Inverse hyperbolic cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcoshOp;

/// Exponential-minus-one operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Expm1Op;

/// Logarithm-of-one-plus operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Log1pOp;

/// Sign operation marker, returning `-1`, `0`, or `1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SignOp;

/// Floor operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct FloorOp;

/// Ceiling operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CeilOp;

/// Round-to-nearest-even operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoundOp;

/// Truncation-toward-zero operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TruncOp;

/// Gauss error function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErfOp;

/// Complementary Gauss error function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErfcOp;

/// Natural logarithm of the absolute gamma function operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LgammaOp;

/// Exact Gaussian Error Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluOp;

/// Exact Gaussian Error Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluGradOp;

/// Rectified linear unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluOp;

/// Rectified linear unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReluGradOp;

/// Logistic sigmoid operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidOp;

/// Logistic sigmoid gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SigmoidGradOp;

/// Hyperbolic tangent operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhOp;

/// Hyperbolic tangent gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhGradOp;

/// Tanh-approximated Gaussian Error Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhOp;

/// Tanh-approximated Gaussian Error Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluTanhGradOp;

/// Sigmoid Linear Unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluOp;

/// Sigmoid Linear Unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluGradOp;

/// Softplus operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusOp;

/// Softplus gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftplusGradOp;

/// Mish activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MishOp;

/// Mish activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MishGradOp;

/// Exponential linear unit operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct EluOp;

/// Exponential linear unit gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct EluGradOp;

/// Hard sigmoid activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardsigmoidOp;

/// Hard sigmoid activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardsigmoidGradOp;

/// Hard swish activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardswishOp;

/// Hard swish activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardswishGradOp;

/// Softsign activation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftsignOp;

/// Softsign activation gradient operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftsignGradOp;

impl UnaryExpr<Wgsl> for ExpOp {
    const EXPR: &'static str = "exp(x)";
}
impl UnaryExpr<CudaC> for ExpOp {
    const EXPR: &'static str = "exp(x)";
}

impl UnaryExpr<Wgsl> for ExpNegOp {
    const EXPR: &'static str = "exp(-x)";
}
impl UnaryExpr<CudaC> for ExpNegOp {
    const EXPR: &'static str = "exp(-x)";
}

impl UnaryExpr<Wgsl> for LnOp {
    const EXPR: &'static str = "log(x)";
}
impl UnaryExpr<CudaC> for LnOp {
    const EXPR: &'static str = "log(x)";
}

impl UnaryExpr<Wgsl> for SinOp {
    const EXPR: &'static str = "sin(x)";
}
impl UnaryExpr<CudaC> for SinOp {
    const EXPR: &'static str = "sin(x)";
}

impl UnaryExpr<Wgsl> for CosOp {
    const EXPR: &'static str = "cos(x)";
}
impl UnaryExpr<CudaC> for CosOp {
    const EXPR: &'static str = "cos(x)";
}

impl UnaryExpr<Wgsl> for SqrtOp {
    const EXPR: &'static str = "sqrt(x)";
}
impl UnaryExpr<CudaC> for SqrtOp {
    const EXPR: &'static str = "sqrt(x)";
}

impl UnaryExpr<Wgsl> for AbsOp {
    const EXPR: &'static str = "abs(x)";
}
impl UnaryExpr<CudaC> for AbsOp {
    const EXPR: &'static str = "abs(x)";
}

impl UnaryExpr<Wgsl> for NegOp {
    const EXPR: &'static str = "-x";
}
impl UnaryExpr<CudaC> for NegOp {
    const EXPR: &'static str = "-x";
}

impl UnaryExpr<Wgsl> for RecipOp {
    const EXPR: &'static str = "1.0 / x";
}
impl UnaryExpr<CudaC> for RecipOp {
    const EXPR: &'static str = "1.0 / x";
}

impl UnaryExpr<Wgsl> for IdentityOp {
    const EXPR: &'static str = "x";
}
impl UnaryExpr<CudaC> for IdentityOp {
    const EXPR: &'static str = "x";
}

macro_rules! impl_math_unary_exprs {
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

impl_math_unary_exprs!(
    (TanOp, "tan(x)", "tan(x)"),
    (AsinOp, "asin(x)", "asin(x)"),
    (AcosOp, "acos(x)", "acos(x)"),
    (AtanOp, "atan(x)", "atan(x)"),
    (SinhOp, "sinh(x)", "sinh(x)"),
    (CoshOp, "cosh(x)", "cosh(x)"),
    (Log2Op, "log2(x)", "log2(x)"),
    (
        Log10Op,
        "log(x) * 0.43429448190325182f",
        "log(x) * 0.43429448190325182f"
    ),
    (Exp2Op, "exp2(x)", "exp2(x)"),
    (AtanhOp, "atanh(x)", "atanh(x)"),
    (AsinhOp, "asinh(x)", "asinh(x)"),
    (AcoshOp, "acosh(x)", "acosh(x)"),
    (Expm1Op, "(exp(x) - 1.0)", "(exp(x) - 1.0f)"),
    (Log1pOp, "log(1.0 + (x))", "log(1.0f + (x))"),
    (
        SignOp,
        "select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0)",
        "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)"
    ),
    (FloorOp, "floor(x)", "floor(x)"),
    (CeilOp, "ceil(x)", "ceil(x)"),
    (RoundOp, "round(x)", "rint(x)"),
    (TruncOp, "trunc(x)", "trunc(x)"),
);

macro_rules! impl_hip_unary_exprs {
    ($(($op:ty, $expr:literal)),+ $(,)?) => {
        $(
            impl UnaryExpr<HipC> for $op {
                const EXPR: &'static str = $expr;
            }
        )+
    };
}

impl_hip_unary_exprs!(
    (ExpOp, "exp(x)"),
    (ExpNegOp, "exp(-x)"),
    (LnOp, "log(x)"),
    (SinOp, "sin(x)"),
    (CosOp, "cos(x)"),
    (SqrtOp, "sqrt(x)"),
    (AbsOp, "abs(x)"),
    (NegOp, "-x"),
    (RecipOp, "1.0 / x"),
    (IdentityOp, "x"),
    (TanOp, "tan(x)"),
    (AsinOp, "asin(x)"),
    (AcosOp, "acos(x)"),
    (AtanOp, "atan(x)"),
    (SinhOp, "sinh(x)"),
    (CoshOp, "cosh(x)"),
    (Log2Op, "log2(x)"),
    (Log10Op, "log(x) * 0.43429448190325182f"),
    (Exp2Op, "exp2(x)"),
    (AtanhOp, "atanh(x)"),
    (AsinhOp, "asinh(x)"),
    (AcoshOp, "acosh(x)"),
    (Expm1Op, "(exp(x) - 1.0f)"),
    (Log1pOp, "log(1.0f + (x))"),
    (SignOp, "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)"),
    (FloorOp, "floor(x)"),
    (CeilOp, "ceil(x)"),
    (RoundOp, "rint(x)"),
    (TruncOp, "trunc(x)"),
    (ErfOp, "erf(x)"),
    (ErfcOp, "erfc(x)"),
    (LgammaOp, "lgamma(x)"),
    (GeluOp, "0.5f * x * (1.0f + erff(x * 0.7071067811865476f))"),
    (
        GeluGradOp,
        "0.5f * (1.0f + erff(x * 0.7071067811865476f)) + x * expf(-0.5f * x * x) * 0.3989422804014327f"
    ),
    (ReluOp, "max(x, 0.0f)"),
    (ReluGradOp, "x > 0.0f ? 1.0f : 0.0f"),
    (SigmoidOp, "1.0f / (1.0f + exp(-x))"),
    (SigmoidGradOp, "x * (1.0f - x)"),
    (TanhOp, "tanh(x)"),
    (TanhGradOp, "1.0f - x * x"),
    (
        GeluTanhOp,
        "0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
    ),
    (
        GeluTanhGradOp,
        "0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x))) + 0.5f * x * (1.0f - tanh(0.7978845608f * (x + 0.044715f * x * x * x)) * tanh(0.7978845608f * (x + 0.044715f * x * x * x))) * 0.7978845608f * (1.0f + 0.134145f * x * x)"
    ),
    (SiluOp, "x / (1.0f + exp(-x))"),
    (
        SiluGradOp,
        "(1.0f / (1.0f + exp(-x))) * (1.0f + x * (1.0f - (1.0f / (1.0f + exp(-x)))))"
    ),
    (SoftplusOp, "log(1.0f + exp(x))"),
    (SoftplusGradOp, "1.0f / (1.0f + exp(-x))"),
    (MishOp, "x * tanhf(logf(1.0f + expf(x)))"),
    (
        MishGradOp,
        "tanhf(logf(1.0f + expf(x))) + x * (1.0f - tanhf(logf(1.0f + expf(x))) * tanhf(logf(1.0f + expf(x)))) * (1.0f / (1.0f + expf(-x)))"
    ),
    (EluOp, "x >= 0.0f ? x : expf(x) - 1.0f"),
    (EluGradOp, "x >= 0.0f ? 1.0f : expf(x)"),
    (HardsigmoidOp, "fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)"),
    (
        HardsigmoidGradOp,
        "(x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f"
    ),
    (HardswishOp, "x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f"),
    (
        HardswishGradOp,
        "x >= 3.0f ? 1.0f : (x > -3.0f ? (2.0f * x + 3.0f) / 6.0f : 0.0f)"
    ),
    (SoftsignOp, "x / (1.0f + fabsf(x))"),
    (
        SoftsignGradOp,
        "1.0f / ((1.0f + fabsf(x)) * (1.0f + fabsf(x)))"
    ),
);
