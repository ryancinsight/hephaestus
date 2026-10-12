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

/// Unnormalized `sinc` operation marker: `sin(x)/x` with `sinc(0) = 1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SincOp;

/// Bessel function of the first kind J0 operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct J0Op;

/// Bessel function of the first kind J1 operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct J1Op;

/// Modified Bessel function of the second kind K0 operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct K0Op;

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
        // No `f` suffix: the AbstractFloat coefficient converts to the
        // operand type, so one spelling serves `f32` (identical rounding to
        // the suffixed literal) and `f64`. A suffixed `f32` literal would
        // poison the `f64` multiply, which has no implicit conversion.
        "log(x) * 0.43429448190325182",
        "log(x) * 0.43429448190325182f"
    ),
    (Exp2Op, "exp2(x)", "exp2(x)"),
    (AtanhOp, "atanh(x)", "atanh(x)"),
    (AsinhOp, "asinh(x)", "asinh(x)"),
    (AcoshOp, "acosh(x)", "acosh(x)"),
    (FloorOp, "floor(x)", "floor(x)"),
    (CeilOp, "ceil(x)", "ceil(x)"),
    (RoundOp, "round(x)", "rint(x)"),
    (TruncOp, "trunc(x)", "trunc(x)"),
);

// Expm1 and Log1p stay out of the macro: WGSL has neither builtin, so near
// zero they evaluate a polynomial core (no `log` call, whose spec floor is
// absolute near 1) while the C dialects call the native 1-ULP
// `expm1`/`log1p` (CUDA Programming Guide Table 17; HIP math API). The C
// spellings stay unsuffixed so overload resolution picks the lane width, as
// `SinOp` does.
//
// The WGSL `expm1` core is the degree-6 Taylor polynomial in Horner form on
// `|x| <= 0.3` (Lagrange remainder below 0.06 ULP there; coefficients are
// exact rationals spelled to 17 digits); outside, `exp(x) - 1` no longer
// cancels. The form is division-free, so only correctly-rounded `+ - *`
// feed the rounding count, and every special input takes an exact arm
// (`±0` through the polynomial, infinities and NaN through `exp`). The
// regrouped `(u - 1) * (x / log(u))` identity was measured first and
// rejected: the Vulkan driver uses the `log` spec floor near 1, failing the
// sweep at 1e7 ULP.
//
// The WGSL `log1p` bands shift into the shared `wgsl_l1p_core` exactly:
// `log(1 + x) = ln 2 + P((x - 1) / 2)` above (Sterbenz-exact argument) and
// `-(ln 2 + P(1 / (2(1 + x)) - 1))` below, so the only `log` calls left
// (`u > 2`, `u < 0.5`) sit under the relative part of the spec table. The
// pole needs no guard (`log(0)` is `-inf`, `log` of a negative is NaN).
macro_rules! wgsl_log1p_expr {
    () => {
        concat!(
            "select(select(select(select(log(1.0 + (x)), (0.0 - (0.6931471805599453 + ",
            wgsl_l1p_core!("((1.0 / (2.0 * (1.0 + (x)))) - 1.0)"),
            ")), (x) >= -0.5), ",
            wgsl_l1p_core!("(x)"),
            ", (x) >= -0.2928932188134525), (0.6931471805599453 + ",
            wgsl_l1p_core!("(((x) - 1.0) * 0.5)"),
            "), (x) > 0.4142135623730951), log(1.0 + (x)), (x) > 1.0)"
        )
    };
}

impl UnaryExpr<Wgsl> for Expm1Op {
    const EXPR: &'static str = "select((exp(x) - 1.0), (x) * (1.0 + (x) * (0.5 + (x) * (0.16666666666666666 + (x) * (0.041666666666666664 + (x) * (0.008333333333333333 + (x) * 0.001388888888888889))))), abs(x) <= 0.3)";
}
impl UnaryExpr<CudaC> for Expm1Op {
    const EXPR: &'static str = "expm1(x)";
}
impl UnaryExpr<Wgsl> for Log1pOp {
    const EXPR: &'static str = wgsl_log1p_expr!();
}
impl UnaryExpr<CudaC> for Log1pOp {
    const EXPR: &'static str = "log1p(x)";
}

// Sign stays out of the macro: the WGSL spelling nests `select` with
// literal-only arms, which concretize to `f32` and reject `f64` output
// buffers, so it reports `SUPPORTS_F64 = false` until a typed unary seam
// lands. The CUDA spelling converts implicitly and serves both precisions.
impl UnaryExpr<Wgsl> for SignOp {
    const EXPR: &'static str = "select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0)";
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for SignOp {
    const EXPR: &'static str = "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)";
}

// Sinc stays out of the macros: the removable singularity branches on exact
// zero (including `-0.0`, which compares equal), so no per-precision epsilon
// literal is needed. The WGSL spelling nests the branch in `select` with
// literal arms, which concretize to `f32` exactly as `SignOp` documents, so
// it reports `SUPPORTS_F64 = false` until a typed unary seam lands (both
// arms evaluate, but the discarded `sin(0)/0` NaN never traps). The C
// spellings convert implicitly and serve both precisions; `sin` stays
// unsuffixed so overload resolution picks the lane width, as `SinOp` does.
impl UnaryExpr<Wgsl> for SincOp {
    const EXPR: &'static str = "select(sin(x) / x, 1.0, x == 0.0)";
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for SincOp {
    const EXPR: &'static str = "((x) == 0.0f ? 1.0f : sin(x) / (x))";
}

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
    (Expm1Op, "expm1(x)"),
    (Log1pOp, "log1p(x)"),
    (SignOp, "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)"),
    (FloorOp, "floor(x)"),
    (CeilOp, "ceil(x)"),
    (RoundOp, "rint(x)"),
    (TruncOp, "trunc(x)"),
    (ErfOp, "erf(x)"),
    (ErfcOp, "erfc(x)"),
    (LgammaOp, "lgamma(x)"),
    (SincOp, "((x) == 0.0f ? 1.0f : sin(x) / (x))"),
    (
        GeluOp,
        "(x) < -3.402823466e+38f ? -0.0f : 0.5f * x * erfcf(-(x * 0.7071067811865476f))"
    ),
    (
        GeluGradOp,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : 0.5f * erfcf(-(x * 0.7071067811865476f)) + x * expf(-0.5f * x * x) * 0.3989422804014327f)"
    ),
    (ReluOp, "max(x, 0.0f)"),
    (ReluGradOp, "x > 0.0f ? 1.0f : 0.0f"),
    (
        SigmoidOp,
        "x <= 0.0f ? exp(x) / (1.0f + exp(x)) : 1.0f / (1.0f + exp(-x))"
    ),
    (SigmoidGradOp, "x * (1.0f - x)"),
    (TanhOp, "tanh(x)"),
    (TanhGradOp, "1.0f - x * x"),
    (SoftplusOp, "x <= 0.0f ? log1p(exp(x)) : x + log1p(exp(-x))"),
    (
        SoftplusGradOp,
        "x <= 0.0f ? exp(x) / (1.0f + exp(x)) : 1.0f / (1.0f + exp(-x))"
    ),
    (EluOp, "x >= 0.0f ? x : expm1(x)"),
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
