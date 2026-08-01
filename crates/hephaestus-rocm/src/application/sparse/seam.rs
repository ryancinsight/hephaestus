//! ROCm/HIP implementation of the device-neutral sparse operator seam.
//!
//! The kernels live in this module's [`spmv`](super::spmv) family; this file
//! only adapts them to [`hephaestus_core::SparseOperatorOps`] so a consumer —
//! or the conformance suite — can run CSR products without naming
//! `RocmDevice`, matching the WGPU and CUDA seam shapes.

use bytemuck::Pod;
use hephaestus_core::{BatchSubmitOps, Result, SparseOperatorOps, StridedView, validate_csr};

use super::prepared::{
    PreparedSparseDispatch, PreparedSpmv, prepare_spmv, submit_prepared_sparse_batch,
};
use super::{GpuCsrMatrix, spmm::spmm_into, spmv::spmv_into};
use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::strided::StridedOperand;
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

    fn apply_batch(
        &self,
        device: &RocmDevice,
        matrix: &Self::Matrix,
        batch: StridedView<'_, RocmBuffer<T>, 2>,
        output: &mut RocmBuffer<T>,
    ) -> Result<()> {
        spmm_into(
            device,
            matrix,
            StridedOperand {
                buffer: batch.buffer,
                layout: batch.layout,
            },
            output,
        )
    }

    fn nnz(&self, matrix: &Self::Matrix) -> usize {
        matrix.nnz()
    }

    type PreparedApply<'op>
        = PreparedSpmv<'op, T>
    where
        T: 'op;

    fn prepare_apply<'op>(
        &self,
        device: &'op RocmDevice,
        matrix: &'op Self::Matrix,
        input: &'op RocmBuffer<T>,
        output: &'op RocmBuffer<T>,
    ) -> Result<Self::PreparedApply<'op>> {
        prepare_spmv(device, matrix, input, output)
    }

    fn dispatch_apply(
        &self,
        _device: &RocmDevice,
        prepared: &Self::PreparedApply<'_>,
    ) -> Result<()> {
        prepared.dispatch()
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

impl<T> BatchSubmitOps<RocmDevice, T> for RocmSparseOps
where
    T: DialectScalar<HipC> + Pod + leto_ops::Scalar,
{
    type Dispatch<'op>
        = PreparedSparseDispatch<'op, 'op, T>
    where
        T: 'op;

    fn spmv_dispatch<'plan, 'op: 'plan>(
        &self,
        prepared: &'plan Self::PreparedApply<'op>,
    ) -> Self::Dispatch<'plan> {
        PreparedSparseDispatch::Spmv(prepared)
    }

    fn submit_batch(&self, _device: &RocmDevice, operations: &[Self::Dispatch<'_>]) -> Result<()> {
        submit_prepared_sparse_batch(operations)
    }
}
