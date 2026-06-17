//! Sparse matrix–vector product `y = A · x` via host delegation.

use super::GpuCsrMatrix;
use crate::application::cuda_type::CudaScalar;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

/// Compute `y = A · x` into a pre-allocated output buffer `y` (length `nrows`).
pub fn spmv_into<T: CudaScalar + leto_ops::Scalar + Pod>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    x: &CudaBuffer<T>,
    y: &mut CudaBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = a.shape();
    if x.len() != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: x.len(),
        });
    }
    if y.len() != nrows {
        return Err(HephaestusError::LengthMismatch {
            host_len: nrows,
            device_len: y.len(),
        });
    }

    let cpu_a = a.to_cpu(device)?;
    let mut host_x = vec![T::ZERO; x.len()];
    device.download(x, &mut host_x)?;

    let layout =
        leto::Layout::c_contiguous([ncols]).map_err(|e| HephaestusError::DispatchFailed {
            message: format!("Invalid layout for x: {e}"),
        })?;
    let view_x = leto::ArrayView1::new(layout, &host_x);

    let mut host_y = vec![T::ZERO; nrows];
    leto_ops::spmv_into(&cpu_a, &view_x, &mut host_y).map_err(|e| {
        HephaestusError::DispatchFailed {
            message: format!("spmv_into failed: {e}"),
        }
    })?;

    device.write_buffer(y, &host_y)
}

/// Compute `y = A · x`, allocating the result buffer.
pub fn spmv<T: CudaScalar + leto_ops::Scalar + Pod>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    x: &CudaBuffer<T>,
) -> Result<CudaBuffer<T>> {
    let (nrows, _) = a.shape();
    let mut y = device.alloc_zeroed::<T>(nrows)?;
    spmv_into(device, a, x, &mut y)?;
    Ok(y)
}
