//! Zero-sized operation markers with per-dialect shader expressions.
//!
//! One op vocabulary for every backend: each marker is a ZST whose dialect
//! expression is an associated const on a
//! [`KernelDialect`](crate::KernelDialect)-parameterized
//! trait, so backend shader templates substitute `Op::EXPR` for their own
//! dialect and dispatch stays fully monomorphized. Consumers add fused ops
//! without touching this crate by implementing the expression trait for
//! their own ZST in the dialects they target — a kernel authored for one
//! dialect does not compile on a backend of another dialect.
//!
//! Canonical operand names (backend templates must bind these locals):
//! - unary expressions read `x`;
//! - binary and combine expressions read `lhs` and `rhs`.

use super::dialect::{CudaC, DialectScalar, KernelDialect, Wgsl};
use bytemuck::Pod;

/// Element expression over the canonical unary operand `x` in dialect `L`.
pub trait UnaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression mapping `x` (e.g. `"exp(-x)"`).
    const EXPR: &'static str;
}

/// Element expression over the canonical operands `lhs`, `rhs` in dialect `L`.
pub trait BinaryExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression combining `lhs` and `rhs` (e.g. `"lhs + rhs"`).
    const EXPR: &'static str;
}

/// Associative combine expression over `lhs`, `rhs` in dialect `L`, used by
/// reductions and scans.
pub trait CombineExpr<L: KernelDialect>: Copy + Send + Sync + 'static {
    /// Expression combining two partial results (e.g. `"max(lhs, rhs)"`).
    const EXPR: &'static str;
}

/// Host-side identity element of op `Op` for this scalar (dialect-free).
pub trait OpIdentity<Op>: Pod {
    /// The identity value (e.g. `0` for sum, `T::MAX` for min).
    const IDENTITY: Self;
}

/// Shader literal token of op `Op`'s identity for this scalar in dialect `L`.
pub trait IdentityToken<Op, L: KernelDialect>: DialectScalar<L> {
    /// The dialect literal (e.g. `"0.0"` in WGSL, `"0.0f"` in CUDA C++).
    const TOKEN: &'static str;
}

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

// ── Binary markers ───────────────────────────────────────────────────────

/// Addition operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AddOp;

/// Subtraction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubOp;

/// Multiplication operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MulOp;

/// Division operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct DivOp;

/// Power operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct PowOp;

impl BinaryExpr<Wgsl> for AddOp {
    const EXPR: &'static str = "lhs + rhs";
}
impl BinaryExpr<CudaC> for AddOp {
    const EXPR: &'static str = "lhs + rhs";
}

impl BinaryExpr<Wgsl> for SubOp {
    const EXPR: &'static str = "lhs - rhs";
}
impl BinaryExpr<CudaC> for SubOp {
    const EXPR: &'static str = "lhs - rhs";
}

impl BinaryExpr<Wgsl> for MulOp {
    const EXPR: &'static str = "lhs * rhs";
}
impl BinaryExpr<CudaC> for MulOp {
    const EXPR: &'static str = "lhs * rhs";
}

impl BinaryExpr<Wgsl> for DivOp {
    const EXPR: &'static str = "lhs / rhs";
}
impl BinaryExpr<CudaC> for DivOp {
    const EXPR: &'static str = "lhs / rhs";
}

impl BinaryExpr<Wgsl> for PowOp {
    const EXPR: &'static str = "pow(lhs, rhs)";
}
impl BinaryExpr<CudaC> for PowOp {
    const EXPR: &'static str = "pow(lhs, rhs)";
}

// ── Reduction markers ────────────────────────────────────────────────────

/// Sum-reduction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumOp;

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

// ── Identities ───────────────────────────────────────────────────────────
// Host values are dialect-free; literal tokens differ per dialect (WGSL has
// no `f` suffix, CUDA C++ float literals carry one so arithmetic stays in
// `float` rather than promoting to `double`).

impl OpIdentity<SumOp> for f32 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<SumOp> for u32 {
    const IDENTITY: Self = 0;
}
impl OpIdentity<SumOp> for i32 {
    const IDENTITY: Self = 0;
}

impl OpIdentity<MinOp> for f32 {
    const IDENTITY: Self = f32::MAX;
}
impl OpIdentity<MinOp> for u32 {
    const IDENTITY: Self = u32::MAX;
}
impl OpIdentity<MinOp> for i32 {
    const IDENTITY: Self = i32::MAX;
}

impl OpIdentity<MaxOp> for f32 {
    const IDENTITY: Self = f32::MIN;
}
impl OpIdentity<MaxOp> for u32 {
    const IDENTITY: Self = u32::MIN;
}
impl OpIdentity<MaxOp> for i32 {
    const IDENTITY: Self = i32::MIN;
}

impl OpIdentity<CumSumOp> for f32 {
    const IDENTITY: Self = 0.0;
}
impl OpIdentity<CumSumOp> for u32 {
    const IDENTITY: Self = 0;
}
impl OpIdentity<CumSumOp> for i32 {
    const IDENTITY: Self = 0;
}

impl OpIdentity<CumProdOp> for f32 {
    const IDENTITY: Self = 1.0;
}
impl OpIdentity<CumProdOp> for u32 {
    const IDENTITY: Self = 1;
}
impl OpIdentity<CumProdOp> for i32 {
    const IDENTITY: Self = 1;
}

impl IdentityToken<SumOp, Wgsl> for f32 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<SumOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<SumOp, Wgsl> for i32 {
    const TOKEN: &'static str = "0";
}
impl IdentityToken<SumOp, CudaC> for f32 {
    const TOKEN: &'static str = "0.0f";
}
impl IdentityToken<SumOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<SumOp, CudaC> for i32 {
    const TOKEN: &'static str = "0";
}

impl IdentityToken<MinOp, Wgsl> for f32 {
    const TOKEN: &'static str = "3.402823466e+38";
}
impl IdentityToken<MinOp, Wgsl> for u32 {
    const TOKEN: &'static str = "4294967295u";
}
impl IdentityToken<MinOp, Wgsl> for i32 {
    const TOKEN: &'static str = "2147483647";
}
impl IdentityToken<MinOp, CudaC> for f32 {
    const TOKEN: &'static str = "3.402823466e+38f";
}
impl IdentityToken<MinOp, CudaC> for u32 {
    const TOKEN: &'static str = "4294967295u";
}
impl IdentityToken<MinOp, CudaC> for i32 {
    const TOKEN: &'static str = "2147483647";
}

impl IdentityToken<MaxOp, Wgsl> for f32 {
    const TOKEN: &'static str = "-3.402823466e+38";
}
impl IdentityToken<MaxOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<MaxOp, Wgsl> for i32 {
    const TOKEN: &'static str = "-2147483648";
}
impl IdentityToken<MaxOp, CudaC> for f32 {
    const TOKEN: &'static str = "-3.402823466e+38f";
}
impl IdentityToken<MaxOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<MaxOp, CudaC> for i32 {
    const TOKEN: &'static str = "-2147483648";
}

impl IdentityToken<CumSumOp, Wgsl> for f32 {
    const TOKEN: &'static str = "0.0";
}
impl IdentityToken<CumSumOp, Wgsl> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<CumSumOp, Wgsl> for i32 {
    const TOKEN: &'static str = "0";
}
impl IdentityToken<CumSumOp, CudaC> for f32 {
    const TOKEN: &'static str = "0.0f";
}
impl IdentityToken<CumSumOp, CudaC> for u32 {
    const TOKEN: &'static str = "0u";
}
impl IdentityToken<CumSumOp, CudaC> for i32 {
    const TOKEN: &'static str = "0";
}

impl IdentityToken<CumProdOp, Wgsl> for f32 {
    const TOKEN: &'static str = "1.0";
}
impl IdentityToken<CumProdOp, Wgsl> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<CumProdOp, Wgsl> for i32 {
    const TOKEN: &'static str = "1";
}
impl IdentityToken<CumProdOp, CudaC> for f32 {
    const TOKEN: &'static str = "1.0f";
}
impl IdentityToken<CumProdOp, CudaC> for u32 {
    const TOKEN: &'static str = "1u";
}
impl IdentityToken<CumProdOp, CudaC> for i32 {
    const TOKEN: &'static str = "1";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_and_identity_agree_per_dialect() {
        assert_eq!(<SumOp as CombineExpr<Wgsl>>::EXPR, "lhs + rhs");
        assert_eq!(<SumOp as CombineExpr<CudaC>>::EXPR, "lhs + rhs");
        assert_eq!(<f32 as IdentityToken<SumOp, Wgsl>>::TOKEN, "0.0");
        assert_eq!(<f32 as IdentityToken<SumOp, CudaC>>::TOKEN, "0.0f");
        assert_eq!(<f32 as OpIdentity<MinOp>>::IDENTITY, f32::MAX);
        assert_eq!(<u32 as OpIdentity<MaxOp>>::IDENTITY, u32::MIN);
    }

    #[test]
    fn consumer_defined_op_composes_with_the_vocabulary() {
        // A consumer-side fused op: implement the expression trait for a
        // local ZST in the targeted dialect — no substrate changes needed.
        #[derive(Clone, Copy, Debug, Default)]
        struct AffineClampOp;
        impl UnaryExpr<Wgsl> for AffineClampOp {
            const EXPR: &'static str = "clamp(x * 2.0 + 1.0, 0.0, 10.0)";
        }
        fn expr_of<Op: UnaryExpr<Wgsl>>() -> &'static str {
            Op::EXPR
        }
        assert_eq!(
            expr_of::<AffineClampOp>(),
            "clamp(x * 2.0 + 1.0, 0.0, 10.0)"
        );
    }
}
