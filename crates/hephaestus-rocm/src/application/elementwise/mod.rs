//! ROCm elementwise operations over typed device buffers.

use hephaestus_core::{HephaestusError, Result};

use crate::RocmBuffer;

/// Binary elementwise operations.
pub mod binary;
/// Scalar elementwise operations.
pub mod scalar;
/// Unary elementwise operations.
pub mod unary;

pub use binary::{AddOp, DivOp, MulOp, PowOp, SubOp, binary_elementwise, binary_elementwise_into};
pub use scalar::{scalar_elementwise, scalar_elementwise_into};
pub use unary::{
    AbsOp, CosOp, ExpNegOp, ExpOp, IdentityOp, LnOp, NegOp, RecipOp, SinOp, SqrtOp,
    unary_elementwise, unary_elementwise_into,
};

fn reject_output_alias<T, U>(
    input_label: &'static str,
    input: &RocmBuffer<T>,
    out: &RocmBuffer<U>,
) -> Result<()> {
    if input.aliases(out) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("output buffer must not alias {input_label} input"),
        });
    }
    Ok(())
}

fn checked_work_items(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("ROCm elementwise length {len} exceeds the HIP kernel argument range"),
    })
}
