//! Reduction and scan markers with their associative combine expressions.

use super::CombineExpr;
use crate::domain::dialect::{CudaC, HipC, Host, Wgsl};
use eunomia::NumericElement;

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

/// The combine as a value function, the operator's definition (ADR 0061).
///
/// Integer combines wrap in two's complement, the semantics WGSL defines for
/// its kernels; floating-point combines are IEEE-754 operations. `Min` and
/// `Max` return `lhs` unless `rhs` compares strictly less (greater), so a NaN
/// operand never displaces a number already held and a NaN `lhs` is kept.
pub trait CombineValue: Copy + Send + Sync + 'static {
    /// `lhs` combined with `rhs`.
    fn combine<T: NumericElement>(lhs: T, rhs: T) -> T;
}

impl CombineValue for SumOp {
    fn combine<T: NumericElement>(lhs: T, rhs: T) -> T {
        lhs.wrapping_add(rhs)
    }
}

impl CombineValue for ProdOp {
    fn combine<T: NumericElement>(lhs: T, rhs: T) -> T {
        lhs.wrapping_mul(rhs)
    }
}

impl CombineValue for MinOp {
    fn combine<T: NumericElement>(lhs: T, rhs: T) -> T {
        if rhs < lhs { rhs } else { lhs }
    }
}

impl CombineValue for MaxOp {
    fn combine<T: NumericElement>(lhs: T, rhs: T) -> T {
        if rhs > lhs { rhs } else { lhs }
    }
}

// ── Scan markers ─────────────────────────────────────────────────────────

/// Cumulative-sum scan operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumSumOp;

/// Cumulative-product scan operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumProdOp;

impl CombineValue for CumSumOp {
    fn combine<T: NumericElement>(lhs: T, rhs: T) -> T {
        SumOp::combine(lhs, rhs)
    }
}

impl CombineValue for CumProdOp {
    fn combine<T: NumericElement>(lhs: T, rhs: T) -> T {
        ProdOp::combine(lhs, rhs)
    }
}

/// Every operator with a value function is a host combine. The overlap
/// analysis (ADR 0061 Consequences): a downstream `impl CombineExpr<Host>`
/// for a local type without `CombineValue` stays legal and reports `None`
/// from [`CombineExpr::value`]; per-dialect impls for other dialects never
/// overlap.
impl<Op: CombineValue> CombineExpr<Host> for Op {
    const EXPR: &'static str = "host";

    fn value<T: NumericElement>(lhs: T, rhs: T) -> Option<T> {
        Some(Op::combine(lhs, rhs))
    }
}

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
