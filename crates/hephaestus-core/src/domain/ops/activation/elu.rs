//! ELU expression.

use super::super::EluOp;
use super::super::UnaryExpr;
use super::super::unary_math::wgsl_expm1_expr;
use crate::domain::dialect::{CudaC, Wgsl};

// Elu stays out of the macro (KERNEL-TAIL): the naive `exp(x) - 1`
// negative arm cancels to `0` near `0-`. The arm is exactly `expm1`,
// so the C spelling calls the native builtin under a ternary gate (the
// `SincOp` precedent, so lane promotion keeps f64 served) and the WGSL
// spelling nests the whole `expm1` select under the `x >= 0` gate,
// reusing its Taylor coefficients verbatim instead of duplicating them.
// Both `select` arms evaluate (the discarded `exp(x)` overflows to
// `+inf` for large positive `x` without trapping, the `SincOp`
// precedent), and the nested literal-arm selects concretize `f32`, so
// the spelling reports `SUPPORTS_F64 = false` per the `Expm1Op`
// precedent. Accuracy is the slice-1 arm proof: the outer select is
// exact, and the inner arm holds under 5e-7 relative (20x inside the
// clause's `1e-5`).
impl UnaryExpr<Wgsl> for EluOp {
    const EXPR: &'static str = concat!("select(", wgsl_expm1_expr!(), ", x, x >= 0.0)");
    const SUPPORTS_F64: bool = false;
}
impl UnaryExpr<CudaC> for EluOp {
    const EXPR: &'static str = "((x) >= 0.0f ? (x) : expm1(x))";
}
