//! Shared validation for ROCm decomposition operands.

use bytemuck::Pod;
use hephaestus_core::{DeviceBuffer, Result, require_dense_operand, validate_square_operand};

use crate::application::strided::StridedOperand;

/// Validate a rank-2 square operand and return its dimension.
pub(crate) fn validate_square<T: Pod>(matrix: &StridedOperand<'_, T, 2>) -> Result<usize> {
    validate_square_operand(matrix.layout, matrix.buffer.len())
}

/// Require a dense C-contiguous zero-offset operand for the blocked entry
/// point, whose contract bulk-copies the source storage on the device.
pub(crate) fn validate_dense_operand<T: Pod>(
    label: &str,
    matrix: &StridedOperand<'_, T, 2>,
) -> Result<()> {
    require_dense_operand(label, matrix.layout)
}
