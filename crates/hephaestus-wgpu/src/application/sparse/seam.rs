//! WGPU implementation of the device-neutral sparse operator seam.

use bytemuck::Pod;
use hephaestus_core::{Result, SparseOperatorOps, StridedView, validate_csr};

use super::spmv::PreparedSpmv;
use super::{GpuCsrMatrix, spmm::spmm_into, spmv::prepare_spmv, spmv::spmv_into};
use crate::application::strided::StridedOperand;
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

    fn apply_batch(
        &self,
        device: &WgpuDevice,
        matrix: &Self::Matrix,
        batch: StridedView<'_, WgpuBuffer<T>, 2>,
        output: &mut WgpuBuffer<T>,
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

    type PreparedApply<'op>
        = PreparedSpmv<T>
    where
        T: 'op;

    fn prepare_apply<'op>(
        &self,
        device: &'op WgpuDevice,
        matrix: &'op Self::Matrix,
        input: &'op WgpuBuffer<T>,
        output: &'op WgpuBuffer<T>,
    ) -> Result<Self::PreparedApply<'op>> {
        prepare_spmv(device, matrix, input, output)
    }

    fn dispatch_apply(
        &self,
        _device: &WgpuDevice,
        prepared: &Self::PreparedApply<'_>,
    ) -> Result<()> {
        prepared.dispatch();
        Ok(())
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
