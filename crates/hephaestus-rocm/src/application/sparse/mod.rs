//! GPU-resident Compressed Sparse Row (CSR) matrices and HIP dispatch.

mod spmm;
mod spmv;

pub use spmm::{spmm, spmm_into, spmv_many, spmv_many_into};
pub use spmv::{spmv, spmv_into};

use crate::RocmDevice;
use crate::infrastructure::RocmBuffer;
use bytemuck::{Pod, Zeroable};
use hephaestus_core::{ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result};

/// Compressed Sparse Row matrix stored in ROCm device buffers.
///
/// The values, column indices, and row pointers remain device-resident after
/// [`Self::from_cpu`]. The HIP kernels consume this representation directly,
/// so sparse dispatch does not materialize a dense matrix.
#[derive(Debug)]
pub struct GpuCsrMatrix<T> {
    values: RocmBuffer<T>,
    col_indices: RocmBuffer<u32>,
    row_ptr: RocmBuffer<u32>,
    nrows: usize,
    ncols: usize,
}

impl<T: DialectScalar<HipC> + Pod + leto_ops::Scalar> GpuCsrMatrix<T> {
    /// Upload a CPU-side Leto CSR matrix to ROCm device storage.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error if an index exceeds the HIP `u32` index
    /// contract, or a transfer/allocation error from the ROCm device.
    pub fn from_cpu(device: &RocmDevice, cpu_matrix: &leto_ops::CsrMatrix<T>) -> Result<Self> {
        let (values, col_indices, row_ptr) = cpu_matrix.as_parts();
        let (nrows, ncols) = cpu_matrix.shape();
        let col_indices = col_indices
            .iter()
            .copied()
            .map(|index| {
                u32::try_from(index).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CSR column index {index} exceeds u32 range"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let row_ptr = row_ptr
            .iter()
            .copied()
            .map(|index| {
                u32::try_from(index).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CSR row pointer {index} exceeds u32 range"),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            values: device.upload(values)?,
            col_indices: device.upload(&col_indices)?,
            row_ptr: device.upload(&row_ptr)?,
            nrows,
            ncols,
        })
    }

    /// Download this ROCm CSR matrix into a CPU-side Leto CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns a transfer error or a typed reconstruction error when the
    /// downloaded CSR metadata cannot satisfy the Leto invariants.
    pub fn to_cpu(&self, device: &RocmDevice) -> Result<leto_ops::CsrMatrix<T>> {
        let mut values = vec![T::zeroed(); self.values.len()];
        device.download(&self.values, &mut values)?;

        let mut col_indices_u32 = vec![0_u32; self.col_indices.len()];
        device.download(&self.col_indices, &mut col_indices_u32)?;
        let col_indices = col_indices_u32
            .into_iter()
            .map(|index| {
                usize::try_from(index).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CSR column index {index} cannot fit usize"),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut row_ptr_u32 = vec![0_u32; self.row_ptr.len()];
        device.download(&self.row_ptr, &mut row_ptr_u32)?;
        let row_ptr = row_ptr_u32
            .into_iter()
            .map(|index| {
                usize::try_from(index).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CSR row pointer {index} cannot fit usize"),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        leto_ops::CsrMatrix::from_parts(values, col_indices, row_ptr, self.nrows, self.ncols)
            .map_err(|error| HephaestusError::AllocationFailed {
                message: format!("failed to reconstruct CPU CSR matrix: {error}"),
            })
    }

    /// Return `(rows, columns)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Return the number of stored non-zero entries.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Borrow the device-resident CSR values.
    #[must_use]
    #[inline]
    pub(crate) fn values(&self) -> &RocmBuffer<T> {
        &self.values
    }

    /// Borrow the device-resident CSR column indices.
    #[must_use]
    #[inline]
    pub(crate) fn col_indices(&self) -> &RocmBuffer<u32> {
        &self.col_indices
    }

    /// Borrow the device-resident CSR row pointers.
    #[must_use]
    #[inline]
    pub(crate) fn row_ptr(&self) -> &RocmBuffer<u32> {
        &self.row_ptr
    }
}
