//! Decomposition operand validation — thin adapters over the backend-neutral
//! `hephaestus_core` validators.

use bytemuck::Pod;
#[cfg(feature = "cuda")]
use hephaestus_core::require_dense_operand;
use hephaestus_core::{DeviceBuffer, Result, validate_square_operand};

use crate::application::strided::StridedOperand;

/// Validate that the input is a square matrix and return its dimension.
pub(crate) fn validate_square<T: Pod>(matrix: &StridedOperand<'_, T, 2>) -> Result<usize> {
    validate_square_operand(matrix.layout, matrix.buffer.len())
}

/// Require a dense C-contiguous zero-offset operand for a blocked decomposition.
#[cfg(feature = "cuda")]
pub(crate) fn validate_dense_operand<T: Pod>(
    label: &str,
    matrix: &StridedOperand<'_, T, 2>,
) -> Result<()> {
    require_dense_operand(label, matrix.layout)
}
