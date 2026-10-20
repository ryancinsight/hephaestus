//! HIP C expressions for the unary markers.

use super::UnaryExpr;
use super::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, EluGradOp,
    EluOp, ErfOp, ErfcOp, Exp2Op, ExpNegOp, ExpOp, Expm1Op, FloorOp, GeluGradOp, GeluOp,
    HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp, IdentityOp, LgammaOp, LnOp,
    Log1pOp, Log2Op, Log10Op, NegOp, RecipOp, ReluGradOp, ReluOp, RoundOp, SigmoidGradOp,
    SigmoidOp, SignOp, SinOp, SincOp, SinhOp, SoftplusGradOp, SoftplusOp, SoftsignGradOp,
    SoftsignOp, SqrtOp, TanOp, TanhGradOp, TanhOp, TruncOp,
};
use crate::domain::dialect::HipC;

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
    // Native builtins, unsuffixed for lane-width overload (see above).
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
    // Native `erfc` (see activation.rs); the unsuffixed builtin keeps f64 served.
    (
        GeluOp,
        "(x) < -3.402823466e+38f ? -0.0f : 0.5f * x * erfc(-x * 0.7071067811865476f)"
    ),
    // Native `erfc` (see activation.rs); the unsuffixed builtins keep f64 served.
    (
        GeluGradOp,
        "(x) > 3.402823466e+38f ? 1.0f : ((x) < -3.402823466e+38f ? 0.0f : 0.5f * erfc(-x * 0.7071067811865476f) + x * exp(-0.5f * x * x) * 0.3989422804014327f)"
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
    // GeluTanhOp, GeluTanhGradOp, SiluOp, and SiluGradOp live in
    // `activation/stable.rs`: macro-built stabilized forms.
    // Stable `max + log1p` form (see activation.rs); native builtins keep f64.
    (
        SoftplusOp,
        "((x) > 0.0f ? (x) : 0.0f) + log1p(exp(-fabs(x)))"
    ),
    (
        SoftplusGradOp,
        "x <= 0.0f ? exp(x) / (1.0f + exp(x)) : 1.0f / (1.0f + exp(-x))"
    ),
    // MishOp and MishGradOp live in `activation/stable.rs`: macro-built
    // stabilized forms.
    // Native `expm1` on the negative arm (see activation.rs); the unsuffixed
    // builtin keeps f64 served.
    (EluOp, "((x) >= 0.0f ? (x) : expm1(x))"),
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
