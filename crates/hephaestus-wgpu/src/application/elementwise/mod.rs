//! Elementwise compute operations.

/// Binary elementwise compute operations.
pub mod binary;
/// Scalar elementwise compute operations.
pub mod scalar;
/// Unary elementwise compute operations.
pub mod unary;

pub use binary::{binary_elementwise, binary_elementwise_into, AddOp, BinaryWgslOp, MulOp, SubOp};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    unary_elementwise, unary_elementwise_into, AbsOp, CosOp, ExpOp, LnOp, NegOp, RecipOp, SinOp,
    SqrtOp, UnaryWgslOp,
};
