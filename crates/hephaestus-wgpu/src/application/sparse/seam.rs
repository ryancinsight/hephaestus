//! WGPU implementation of the device-neutral sparse operator seam.

use bytemuck::Pod;
use hephaestus_core::{Result, SparseOperatorOps, validate_csr};

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
