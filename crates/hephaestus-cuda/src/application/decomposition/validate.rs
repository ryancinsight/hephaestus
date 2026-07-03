//! Shared validation utilities for decompositions.

use bytemuck::Pod;
use hephaestus_core::{DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};

/// Validate that the input is a square matrix and return its dimension.
pub(crate) fn validate_square<T: Pod>(matrix: &StridedOperand<'_, T, 2>) -> Result<usize> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Decomposition requires a square matrix, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    Ok(rows)
}

/// Require a dense C-contiguous zero-offset operand.
///
/// The blocked decomposition entry points bulk-copy `rows·cols` elements
/// straight from `matrix.buffer.raw()`. A transposed, offset, or broadcast
/// (zero-stride) layout would make that raw copy compute from the wrong
/// elements — and for layouts whose validated storage extent is smaller than
/// `rows·cols` (broadcast), read past the allocation. `validate_storage_len`
/// only bounds the layout's own extent, so density is checked explicitly
/// here before any raw copy.
pub(crate) fn validate_dense_operand<T: Pod>(
    label: &str,
    matrix: &StridedOperand<'_, T, 2>,
) -> Result<()> {
    if matrix.layout.is_c_contiguous() {
        return Ok(());
    }
    Err(HephaestusError::DispatchFailed {
        message: format!(
            "{label} blocked decomposition requires a dense C-contiguous zero-offset operand; \
             got shape {:?}, strides {:?}, offset {} — materialize the view first (e.g. an \
             identity strided copy into a fresh buffer)",
            matrix.layout.shape, matrix.layout.strides, matrix.layout.offset
        ),
    })
}
