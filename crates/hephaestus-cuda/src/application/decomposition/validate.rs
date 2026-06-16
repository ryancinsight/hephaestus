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
