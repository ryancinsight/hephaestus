//! Provider-owned sparse operator seam for Metal.
//!
//! Metal delegates wholly to the WGPU implementation, matching
//! [`crate::MetalScanOps`]: the buffer wrapper is unwrapped to its inner WGPU
//! buffer and every method forwards to [`WgpuSparseOps`] on the device's WGPU
//! handle. The device-resident matrix type is WGPU's, since both live on the
//! same underlying device.

use bytemuck::Pod;
use hephaestus_core::{DialectScalar, Result, SparseOperatorOps, StridedView};
use hephaestus_wgpu::{GpuCsrMatrix, MatmulZero, WgpuDevice, WgpuSparseOps, Wgsl};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`SparseOperatorOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalSparseOps {
    inner: WgpuSparseOps,
}

impl<T> SparseOperatorOps<MetalDevice, T> for MetalSparseOps
where
    T: DialectScalar<Wgsl> + MatmulZero + Pod + leto_ops::Scalar,
{
    type Matrix = GpuCsrMatrix<T>;

    fn upload_csr(
        &self,
        device: &MetalDevice,
        values: &[T],
        col_indices: &[usize],
        row_ptr: &[usize],
        rows: usize,
        columns: usize,
    ) -> Result<Self::Matrix> {
        self.inner.upload_csr(
            device.wgpu_device(),
            values,
            col_indices,
            row_ptr,
            rows,
            columns,
        )
    }

    fn shape(&self, matrix: &Self::Matrix) -> (usize, usize) {
        <WgpuSparseOps as SparseOperatorOps<WgpuDevice, T>>::shape(&self.inner, matrix)
    }

    fn apply_batch(
        &self,
        device: &MetalDevice,
        matrix: &Self::Matrix,
        batch: StridedView<'_, MetalBuffer<T>, 2>,
        output: &mut MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.apply_batch(
            device.wgpu_device(),
            matrix,
            StridedView::new(&batch.buffer.inner, batch.layout),
            &mut output.inner,
        )
    }

    fn nnz(&self, matrix: &Self::Matrix) -> usize {
        <WgpuSparseOps as SparseOperatorOps<WgpuDevice, T>>::nnz(&self.inner, matrix)
    }

    fn apply(
        &self,
        device: &MetalDevice,
        matrix: &Self::Matrix,
        input: &MetalBuffer<T>,
        output: &mut MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.apply(
            device.wgpu_device(),
            matrix,
            &input.inner,
            &mut output.inner,
        )
    }
}
