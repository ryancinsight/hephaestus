//! Elementwise CUDA compute operations.

use crate::infrastructure::buffer::CudaBuffer;
use hephaestus_core::{HephaestusError, Result};

/// Binary elementwise operations.
pub mod binary;
/// Scalar elementwise operations.
pub mod scalar;
/// Unary elementwise operations.
pub mod unary;

pub use binary::{binary_elementwise, binary_elementwise_into, AddOp, DivOp, MulOp, PowOp, SubOp};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    unary_elementwise, unary_elementwise_into, AbsOp, CosOp, ExpNegOp, ExpOp, IdentityOp, LnOp,
    NegOp, RecipOp, SinOp, SqrtOp,
};

fn reject_output_alias<T, U>(
    input_label: &'static str,
    input: &CudaBuffer<T>,
    out: &CudaBuffer<U>,
) -> Result<()> {
    if input.aliases(out) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("output buffer must not alias {input_label} input"),
        });
    }
    Ok(())
}
