//! Elementwise compute operations.

/// Binary elementwise compute operations.
pub mod binary;
/// Scalar elementwise compute operations.
pub mod scalar;
/// Unary elementwise compute operations.
pub mod unary;

pub use binary::{binary_elementwise, AddOp, BinaryWgslOp, MulOp, SubOp};
pub use scalar::scalar_elementwise;
pub use unary::{
    unary_elementwise, AbsOp, CosOp, ExpOp, LnOp, NegOp, RecipOp, SinOp, SqrtOp, UnaryWgslOp,
};
