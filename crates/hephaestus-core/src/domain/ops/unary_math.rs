//! Wgsl and CUDA expressions for the elementary-math unary markers.

use super::UnaryExpr;
use super::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, Exp2Op,
    ExpNegOp, ExpOp, Expm1Op, FloorOp, IdentityOp, LnOp, Log1pOp, Log2Op, Log10Op, NegOp, RecipOp,
    RoundOp, SignOp, SinOp, SincOp, SinhOp, SqrtOp, TanOp, TruncOp,
};
use crate::domain::dialect::{CudaC, Wgsl};

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
    // Expm1/Log1p stay out of the macro: WGSL has no `expm1`/`log1p`
    // builtin, so the WGSL spellings branch below while the C spellings
    // call the native builtins (see below).
    (FloorOp, "floor(x)", "floor(x)"),
    (CeilOp, "ceil(x)", "ceil(x)"),
    (RoundOp, "round(x)", "rint(x)"),
    (TruncOp, "trunc(x)", "trunc(x)"),
);

// Expm1/Log1p stay out of the macros (KERNEL-TAIL, ADR 0061 Decision 6):
// the naive `(exp(x) - 1.0)` / `log(1.0 + x)` transcriptions cancel to
// exactly `0` once `|x|` reaches f32's `ulp(1)`. The C spellings call the
// native builtins unsuffixed, so overload resolution picks the lane width
// exactly as `SinOp` documents and both precisions stay served. WGSL has
// neither builtin, so both WGSL spellings select a tail-accurate small
// arm (`|x| < 0.5` for expm1, `|x| < 0.15` for log1p) and the naive
// transcription above, where `u = exp(x)` (resp. `u = 1 + x`) sits far
// enough from 1 that the subtraction is already accurate — and
// bit-identical to the previous rendering, so the fix is strictly
// tail-local.
//
// The expm1 small arm is the degree-8 Taylor series in Horner form: the
// truncation remainder `|x|^9 / 9!` peaks at `1.1e-8` relative on
// `|x| <= 0.5`, and eight roundings land the arm under 5e-7 relative,
// inside the conformance clause's `1e-6`. The degree is f32-graded (an
// f64 arm needs degree 15 for the same crossover), and the `u - 1` arm
// calls `exp`, which the WGSL canary records as f32-grade at f64 on the
// reference adapter — so the spelling reports `SUPPORTS_F64 = false`
// until an f64 sweep proves a degree-bumped form.
//
// The log1p small arm is `x * g(x)` where `g` is the degree-5 Chebyshev
// interpolant of `log1p(x) / x` on `|x| <= 0.15` (second-kind nodes,
// monomial basis solved in f64), in Horner form: a dense f32 sweep of
// the exact op sequence peaks at 2.4e-7 relative (4.2x inside the
// clause's `1e-6`), and the `x = -1` lane stays on the `log(u)` arm
// (`log(0) = -inf`, matching `ln_1p`). A compensated `log(u) -
// (u - 1 - x) / u` arm was tried first and reads back `0` on-device: the
// toolchain folds the algebraically-zero correction (proven by sentinel
// probe on the reference adapter), so WGSL tail arms must be pure
// polynomials with no cancellable subexpression — the expm1 Taylor arm
// above survives for exactly that reason. The spelling nests `select`
// and reports `SUPPORTS_F64 = false` per the `SignOp` precedent.
impl UnaryExpr<Wgsl> for Expm1Op {
    const EXPR: &'static str = "select(exp(x) - 1.0, x * (1.0 + x * (0.5 + x * (0.16666666666666666 + x * (0.041666666666666664 + x * (0.008333333333333333 + x * (0.001388888888888889 + x * (0.0001984126984126984 + x * 0.0000248015873015873))))))), abs(x) < 0.5)";
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for Expm1Op {
    const EXPR: &'static str = "expm1(x)";
}
impl UnaryExpr<Wgsl> for Log1pOp {
    const EXPR: &'static str = "select(log(1.0 + x), (x * (1.00000010489199531 + x * ((-5.00000091861519613e-1) + x * (3.33272810783040052e-1 + x * ((-2.49946998041162383e-1) + x * (2.05754288391658408e-1 + x * (-1.71704947320801854e-1))))))), abs(x) < 0.15)";
    const SUPPORTS_F64: bool = false;
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
