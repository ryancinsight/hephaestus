//! CUDA implementation of the device-neutral sparse operator seam.
//!
//! The kernels live in this module's [`spmv`](super::spmv) family; this file
//! only adapts them to [`hephaestus_core::SparseOperatorOps`] so a consumer —
//! or the conformance suite — can run CSR products without naming
//! `CudaDevice`, matching the WGPU seam shape.

use bytemuck::Pod;
use hephaestus_core::{Result, SparseOperatorOps, StridedView, validate_csr};

use super::{GpuCsrMatrix, spmm::spmm_into, spmv::spmv_into};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use hephaestus_core::CudaC;
use hephaestus_core::DialectScalar;

/// Sparse operator support for one CUDA device.
///
/// Zero-sized: the SpMV kernel is cached by the device, so this carries no
/// prepared resources of its own, matching [`crate::CudaVectorOps`]'s role
/// shape.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaSparseOps;

impl<T> SparseOperatorOps<CudaDevice, T> for CudaSparseOps
where
    T: DialectScalar<CudaC> + Pod + leto_ops::Scalar,
{
    type Matrix = GpuCsrMatrix<T>;

    fn upload_csr(
        &self,
        device: &CudaDevice,
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

    fn apply_batch(
        &self,
        device: &CudaDevice,
        matrix: &Self::Matrix,
        batch: StridedView<'_, CudaBuffer<T>, 2>,
        output: &mut CudaBuffer<T>,
    ) -> Result<()> {
        spmm_into(
            device,
            matrix,
            &StridedOperand {
                buffer: batch.buffer,
                layout: batch.layout,
            },
            output,
        )
    }

    fn nnz(&self, matrix: &Self::Matrix) -> usize {
        matrix.nnz()
    }

    fn apply(
        &self,
        device: &CudaDevice,
        matrix: &Self::Matrix,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> Result<()> {
        spmv_into(device, matrix, input, output)
    }
}
