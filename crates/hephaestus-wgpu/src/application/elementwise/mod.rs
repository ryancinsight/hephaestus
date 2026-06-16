//! Elementwise compute operations.

use hephaestus_core::{HephaestusError, Result};

use crate::infrastructure::buffer::WgpuBuffer;

/// Binary elementwise compute operations.
pub mod binary;
/// Scalar elementwise compute operations.
pub mod scalar;
/// Unary elementwise compute operations.
pub mod unary;

pub use binary::{
    binary_elementwise, binary_elementwise_into, AddOp, BinaryWgslOp, DivOp, MulOp, PowOp, SubOp,
};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    unary_elementwise, unary_elementwise_into, AbsOp, CosOp, ExpOp, LnOp, NegOp, RecipOp, SinOp,
    SqrtOp, UnaryWgslOp,
};

fn reject_output_alias<T, U>(
    input_label: &'static str,
    input: &WgpuBuffer<T>,
    out: &WgpuBuffer<U>,
) -> Result<()> {
    if input.aliases(out) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("output buffer must not alias {input_label} input"),
        });
    }
    Ok(())
}
