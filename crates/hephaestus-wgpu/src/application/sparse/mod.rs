//! GPU-resident Compressed Sparse Row (CSR) matrix representation.

mod batch;
mod spmm;
mod spmv;

pub use batch::{PreparedSparseDispatch, submit_prepared_sparse_batch};
pub use spmm::{
    PreparedSpmm, prepare_spmm, prepare_spmv_many, spmm, spmm_into, spmv_many, spmv_many_into,
};
pub use spmv::{PreparedSpmv, prepare_spmv, spmv, spmv_into};

use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, Result, Wgsl};

/// Compressed Sparse Row matrix on the GPU: stores only the non-zero elements
/// in device-resident buffers.
#[derive(Debug, Clone)]
pub struct GpuCsrMatrix<T> {
    values: WgpuBuffer<T>,
    indices: WgpuBuffer<u32>,
    col_indices_len: usize,
    nrows: usize,
    ncols: usize,
}

impl<T: DialectScalar<Wgsl> + Pod + leto_ops::Scalar> GpuCsrMatrix<T> {
    /// Upload a CPU-side Leto `CsrMatrix` to the GPU.
    pub fn from_cpu(device: &WgpuDevice, cpu_matrix: &leto_ops::CsrMatrix<T>) -> Result<Self> {
        let (values, col_indices, row_ptr) = cpu_matrix.as_parts();
        let (nrows, ncols) = cpu_matrix.shape();

        let col_indices_u32 = col_indices
            .iter()
            .map(|&j| {
                u32::try_from(j).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CSR column index {j} exceeds u32 range"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let row_ptr_u32 = row_ptr
            .iter()
            .map(|&r| {
                u32::try_from(r).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CSR row pointer {r} exceeds u32 range"),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut indices = Vec::with_capacity(col_indices_u32.len() + row_ptr_u32.len());
        indices.extend_from_slice(&col_indices_u32);
        indices.extend_from_slice(&row_ptr_u32);

        let values_buf = device.upload(values)?;
        let indices_buf = device.upload(&indices)?;

        Ok(Self {
            values: values_buf,
            indices: indices_buf,
            col_indices_len: col_indices_u32.len(),
            nrows,
            ncols,
        })
    }

    /// Download the GPU-resident sparse matrix back to a CPU Leto `CsrMatrix`.
    pub fn to_cpu(&self, device: &WgpuDevice) -> Result<leto_ops::CsrMatrix<T>> {
        let mut values = vec![T::ZERO; self.values.len()];
        device.download(&self.values, &mut values)?;

        let mut indices = vec![0u32; self.indices.len()];
        device.download(&self.indices, &mut indices)?;

        let (col_u32, row_u32) = indices.split_at(self.col_indices_len);
        let col_indices = col_u32.iter().map(|&j| j as usize).collect::<Vec<_>>();
        let row_ptr = row_u32.iter().map(|&r| r as usize).collect::<Vec<_>>();

        leto_ops::CsrMatrix::from_parts(values, col_indices, row_ptr, self.nrows, self.ncols)
            .map_err(|e| hephaestus_core::HephaestusError::AllocationFailed {
                message: format!("Failed to reconstruct CPU CsrMatrix: {e}"),
            })
    }
}

impl<T> GpuCsrMatrix<T> {
    /// `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of stored non-zero entries.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Access the underlying values buffer.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &WgpuBuffer<T> {
        &self.values
    }

    pub(crate) fn indices(&self) -> &WgpuBuffer<u32> {
        &self.indices
    }

    pub(crate) fn row_ptr_offset(&self) -> usize {
        self.col_indices_len
    }
}
