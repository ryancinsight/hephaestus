//! ROCm/HIP implementation of the device-neutral sparse operator seam.
//!
//! The kernels live in this module's [`spmv`](super::spmv) family; this file
//! only adapts them to [`hephaestus_core::SparseOperatorOps`] so a consumer —
//! or the conformance suite — can run CSR products without naming
//! `RocmDevice`, matching the WGPU and CUDA seam shapes.

use bytemuck::Pod;
use hephaestus_core::{Result, SparseOperatorOps, validate_csr};

use super::{GpuCsrMatrix, spmv::spmv_into};
use crate::RocmBuffer;
use crate::RocmDevice;
use hephaestus_core::DialectScalar;
use hephaestus_core::HipC;

/// Sparse operator support for one ROCm device.
///
/// Zero-sized: the SpMV kernel is cached by the device, so this carries no
/// prepared resources of its own, matching [`crate::RocmVectorOps`]'s role
/// shape.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmSparseOps;

impl<T> SparseOperatorOps<RocmDevice, T> for RocmSparseOps
where
    T: DialectScalar<HipC> + Pod + leto_ops::Scalar,
{
    type Matrix = GpuCsrMatrix<T>;

    fn upload_csr(
        &self,
        device: &RocmDevice,
        values: &[T],
        col_indices: &[usize],
        row_ptr: &[usize],
        rows: usize,
        columns: usize,
    ) -> Result<Self::Matrix> {
        validate_csr(values, col_indices, row_ptr, rows, columns)?;
        let host = leto_ops::CsrMatrix::from_parts(
            values.to_vec(),
            col_indices.to_vec(),
            row_ptr.to_vec(),
            rows,
            columns,
        )
        .map_err(|error| hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("invalid CSR: {error}"),
        })?;
        GpuCsrMatrix::from_cpu(device, &host)
    }

    fn shape(&self, matrix: &Self::Matrix) -> (usize, usize) {
        matrix.shape()
    }

    fn apply(
        &self,
        device: &RocmDevice,
        matrix: &Self::Matrix,
        input: &RocmBuffer<T>,
        output: &mut RocmBuffer<T>,
    ) -> Result<()> {
        spmv_into(device, matrix, input, output)
    }
}
