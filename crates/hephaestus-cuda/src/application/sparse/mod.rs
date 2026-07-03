//! GPU-resident Compressed Sparse Row (CSR) matrix representation for CUDA.

mod spmm;
mod spmv;

pub use spmm::{spmm, spmm_into, spmv_many, spmv_many_into};
pub use spmv::{spmv, spmv_into};

use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, CudaC, DeviceBuffer, DialectScalar, Result};

/// Compressed Sparse Row matrix on the GPU (CUDA): stores only the non-zero elements
/// in device-resident buffers.
#[derive(Debug)]
pub struct GpuCsrMatrix<T> {
    values: CudaBuffer<T>,
    col_indices: CudaBuffer<u32>,
    row_ptr: CudaBuffer<u32>,
    nrows: usize,
    ncols: usize,
}

impl<T: DialectScalar<CudaC> + Pod + leto_ops::Scalar> GpuCsrMatrix<T> {
    /// Upload a CPU-side Leto `CsrMatrix` to the GPU.
    pub fn from_cpu(device: &CudaDevice, cpu_matrix: &leto_ops::CsrMatrix<T>) -> Result<Self> {
        let (values, col_indices, row_ptr) = cpu_matrix.as_parts();
        let (nrows, ncols) = cpu_matrix.shape();

        // Convert usize vectors to u32 for the GPU buffer
        let col_indices_u32 = col_indices.iter().map(|&j| j as u32).collect::<Vec<_>>();
        let row_ptr_u32 = row_ptr.iter().map(|&r| r as u32).collect::<Vec<_>>();

        let values_buf = device.upload(values)?;
        let col_buf = device.upload(&col_indices_u32)?;
        let row_buf = device.upload(&row_ptr_u32)?;

        Ok(Self {
            values: values_buf,
            col_indices: col_buf,
            row_ptr: row_buf,
            nrows,
            ncols,
        })
    }

    /// Download the GPU-resident sparse matrix back to a CPU Leto `CsrMatrix`.
    pub fn to_cpu(&self, device: &CudaDevice) -> Result<leto_ops::CsrMatrix<T>> {
        let mut values = vec![T::ZERO; self.values.len()];
        device.download(&self.values, &mut values)?;

        let mut col_u32 = vec![0u32; self.col_indices.len()];
        device.download(&self.col_indices, &mut col_u32)?;

        let mut row_u32 = vec![0u32; self.row_ptr.len()];
        device.download(&self.row_ptr, &mut row_u32)?;

        let col_indices = col_u32.into_iter().map(|j| j as usize).collect::<Vec<_>>();
        let row_ptr = row_u32.into_iter().map(|r| r as usize).collect::<Vec<_>>();

        leto_ops::CsrMatrix::from_parts(values, col_indices, row_ptr, self.nrows, self.ncols)
            .map_err(|e| hephaestus_core::HephaestusError::AllocationFailed {
                message: format!("Failed to reconstruct CPU CsrMatrix: {e}"),
            })
    }

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
    pub fn values(&self) -> &CudaBuffer<T> {
        &self.values
    }

    /// Access the underlying column indices buffer.
    #[must_use]
    #[inline]
    pub fn col_indices(&self) -> &CudaBuffer<u32> {
        &self.col_indices
    }

    /// Access the underlying row pointers buffer.
    #[must_use]
    #[inline]
    pub fn row_ptr(&self) -> &CudaBuffer<u32> {
        &self.row_ptr
    }
}
