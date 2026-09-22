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

mod activation;
mod activation_value;
mod binary;
mod combine;
mod expr;
mod identity;
mod unary;
mod unary_value;

#[cfg(test)]
mod tests;

pub use expr::{BinaryExpr, CombineExpr, TypedBinaryExpr, UnaryExpr};
pub use identity::{IdentityToken, OpIdentity};

pub use binary::{
    AddOp, BinaryValue, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp, PowOp, SubOp,
    TypedBinaryValue,
};
pub use combine::{CombineValue, CumProdOp, CumSumOp, MaxOp, MinOp, ProdOp, SumOp};
pub use unary::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CeilOp, CosOp, CoshOp, EluGradOp,
    EluOp, ErfOp, ErfcOp, Exp2Op, ExpNegOp, ExpOp, Expm1Op, FloorOp, GeluGradOp, GeluOp,
    GeluTanhGradOp, GeluTanhOp, HardsigmoidGradOp, HardsigmoidOp, HardswishGradOp, HardswishOp,
    IdentityOp, LgammaOp, LnOp, Log1pOp, Log2Op, Log10Op, MishGradOp, MishOp, NegOp, RecipOp,
    ReluGradOp, ReluOp, RoundOp, SigmoidGradOp, SigmoidOp, SignOp, SiluGradOp, SiluOp, SinOp,
    SinhOp, SoftplusGradOp, SoftplusOp, SoftsignGradOp, SoftsignOp, SqrtOp, TanOp, TanhGradOp,
    TanhOp, TruncOp,
};
pub use unary_value::UnaryValue;
