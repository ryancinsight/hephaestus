//! Gradient-mask expressions.

use super::super::UnaryExpr;
use super::super::{HardsigmoidGradOp, ReluGradOp};
use crate::domain::dialect::{CudaC, Wgsl};

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
