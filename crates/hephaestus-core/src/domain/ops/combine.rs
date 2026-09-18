//! Reduction and scan markers with their associative combine expressions.

use super::CombineExpr;
use crate::domain::dialect::{CudaC, HipC, Wgsl};

// ── Reduction markers ────────────────────────────────────────────────────

/// Sum-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumOp;

/// Product-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProdOp;

/// Minimum-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinOp;

/// Maximum-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxOp;

impl CombineExpr<Wgsl> for SumOp {
    const EXPR: &'static str = "lhs + rhs";
}
impl CombineExpr<CudaC> for SumOp {
    const EXPR: &'static str = "lhs + rhs";
}

impl CombineExpr<Wgsl> for ProdOp {
    const EXPR: &'static str = "lhs * rhs";
}
impl CombineExpr<CudaC> for ProdOp {
    const EXPR: &'static str = "lhs * rhs";
}

impl CombineExpr<Wgsl> for MinOp {
    const EXPR: &'static str = "min(lhs, rhs)";
}
impl CombineExpr<CudaC> for MinOp {
    const EXPR: &'static str = "min(lhs, rhs)";
}

impl CombineExpr<Wgsl> for MaxOp {
    const EXPR: &'static str = "max(lhs, rhs)";
}
impl CombineExpr<CudaC> for MaxOp {
    const EXPR: &'static str = "max(lhs, rhs)";
}

// ── Scan markers ─────────────────────────────────────────────────────────

/// Cumulative-sum scan operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumSumOp;

/// Cumulative-product scan operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumProdOp;

impl CombineExpr<Wgsl> for CumSumOp {
    const EXPR: &'static str = "lhs + rhs";
}
impl CombineExpr<CudaC> for CumSumOp {
    const EXPR: &'static str = "lhs + rhs";
}

impl CombineExpr<Wgsl> for CumProdOp {
    const EXPR: &'static str = "lhs * rhs";
}
impl CombineExpr<CudaC> for CumProdOp {
    const EXPR: &'static str = "lhs * rhs";
}

macro_rules! impl_hip_combine_exprs {
    ($(($op:ty, $expr:literal)),+ $(,)?) => {
        $(
            impl CombineExpr<HipC> for $op {
                const EXPR: &'static str = $expr;
            }
        )+
    };
}

impl_hip_combine_exprs!(
    (SumOp, "lhs + rhs"),
    (ProdOp, "lhs * rhs"),
    (MinOp, "min(lhs, rhs)"),
    (MaxOp, "max(lhs, rhs)"),
    (CumSumOp, "lhs + rhs"),
    (CumProdOp, "lhs * rhs"),
);
