//! WGPU implementation of the device-neutral sparse operator seam.

use bytemuck::Pod;
use hephaestus_core::{HephaestusError, Result, SparseOperatorOps};

use super::{GpuCsrMatrix, spmv::spmv_into};
use crate::{MatmulZero, WgpuBuffer, WgpuDevice, Wgsl};
use hephaestus_core::DialectScalar;

/// Sparse operator support for one WGPU device.
///
/// Stateless today: the SpMV pipeline is cached by the device, so this carries
/// no prepared resources of its own. It exists as a value rather than a set of
/// free functions so consumers hold one object per concern, matching
/// [`crate::WgpuVectorOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuSparseOps;

/// Validate canonical CSR structure before anything is uploaded.
///
/// The seam accepts raw parts, so the invariants a host matrix type would have
/// enforced at construction are checked here instead: dispatching against
/// malformed structure would read out of bounds on the device.
fn validate_csr<T>(
    values: &[T],
    col_indices: &[usize],
    row_ptr: &[usize],
    rows: usize,
    columns: usize,
) -> Result<()> {
    let malformed = |reason: String| HephaestusError::DispatchFailed {
        message: format!("invalid CSR: {reason}"),
    };
    if row_ptr.len() != rows + 1 {
        return Err(malformed(format!(
            "row_ptr length {} must be rows + 1 = {}",
            row_ptr.len(),
            rows + 1
        )));
    }
    if col_indices.len() != values.len() {
        return Err(malformed(format!(
            "col_indices length {} differs from values length {}",
            col_indices.len(),
            values.len()
        )));
    }
    if row_ptr[0] != 0 || row_ptr[rows] != values.len() {
        return Err(malformed(
            "row_ptr must start at 0 and end at the nonzero count".to_owned(),
        ));
    }
    for window in row_ptr.windows(2) {
        if window[0] > window[1] {
            return Err(malformed("row_ptr must be non-decreasing".to_owned()));
        }
        let row = &col_indices[window[0]..window[1]];
        if row.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(malformed(
                "column indices in each row must be strictly increasing".to_owned(),
            ));
        }
        if row.last().is_some_and(|&last| last >= columns) {
            return Err(malformed(format!(
                "column index out of range for {columns} columns"
            )));
        }
    }
    Ok(())
}

impl<T> SparseOperatorOps<WgpuDevice, T> for WgpuSparseOps
where
    T: DialectScalar<Wgsl> + MatmulZero + Pod + leto_ops::Scalar,
{
    type Matrix = GpuCsrMatrix<T>;

    fn upload_csr(
        &self,
        device: &WgpuDevice,
        values: &[T],
        col_indices: &[usize],
        row_ptr: &[usize],
        rows: usize,
        columns: usize,
    ) -> Result<Self::Matrix> {
        validate_csr(values, col_indices, row_ptr, rows, columns)?;
        GpuCsrMatrix::from_parts(device, values, col_indices, row_ptr, rows, columns)
    }

    fn shape(&self, matrix: &Self::Matrix) -> (usize, usize) {
        matrix.shape()
    }

    fn apply(
        &self,
        device: &WgpuDevice,
        matrix: &Self::Matrix,
        input: &WgpuBuffer<T>,
        output: &mut WgpuBuffer<T>,
    ) -> Result<()> {
        spmv_into(device, matrix, input, output)
    }
}
