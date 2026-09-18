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

use super::dialect::{DialectScalar, KernelDialect};
use eunomia::Pod;

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

/// Scalar-aware binary expression over the canonical operands `lhs`, `rhs`.
///
/// This seam is required when the result expression depends on the scalar
/// representation. Comparisons are the current example: WGSL and CUDA/HIP
/// require different zero/one literal tokens for floating-point and integer
/// masks. Arithmetic operations should use [`BinaryExpr`] when one expression
/// is valid for every scalar supported by the operation.
pub trait TypedBinaryExpr<L: KernelDialect, T: DialectScalar<L>>:
    Copy + Send + Sync + 'static
{
    /// Expression combining `lhs` and `rhs` for scalar `T` in dialect `L`.
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

mod activation;
mod binary;
mod combine;
mod identity;
mod unary;

#[cfg(test)]
mod tests;

pub use binary::{AddOp, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp, PowOp, SubOp};
pub use combine::{CumProdOp, CumSumOp, MaxOp, MinOp, ProdOp, SumOp};
pub use unary::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, EluGradOp,
    EluOp, ErfOp, ErfcOp, Exp2Op, ExpNegOp, ExpOp, Expm1Op, FloorOp, GeluGradOp, GeluOp,
    GeluTanhGradOp, GeluTanhOp, HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp,
    IdentityOp, LgammaOp, LnOp, Log1pOp, Log2Op, Log10Op, MishGradOp, MishOp, NegOp, RecipOp,
    ReluGradOp, ReluOp, RoundOp, SigmoidGradOp, SigmoidOp, SignOp, SiluGradOp, SiluOp, SinOp,
    SinhOp, SoftplusGradOp, SoftplusOp, SoftsignGradOp, SoftsignOp, SqrtOp, TanOp, TanhGradOp,
    TanhOp, TruncOp,
};
