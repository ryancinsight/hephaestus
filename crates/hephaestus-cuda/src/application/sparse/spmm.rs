//! Sparse–dense matrix product `C = A · B` via host delegation for CUDA.

use super::GpuCsrMatrix;
use crate::application::cuda_type::CudaScalar;
use crate::application::linalg::AsGpuMatrixOperand;
use crate::application::strided::map_layout_err;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

/// Compute `C = A · B` into a pre-allocated output buffer `c`.
pub fn spmm_into<'a, T: CudaScalar + leto_ops::Scalar + Pod, B: AsGpuMatrixOperand<'a, T>>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    b: &B,
    c: &mut CudaBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = a.shape();
    let b_op = b.as_operand();
    let [b_rows, bcols] = b_op.layout.shape;

    if b_rows != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: b_rows,
        });
    }

    let expected_c_len = nrows * bcols;
    if c.len() != expected_c_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: expected_c_len,
            device_len: c.len(),
        });
    }

    let cpu_a = a.to_cpu(device)?;

    b_op.layout
        .validate_storage_len(b_op.buffer.len())
        .map_err(map_layout_err)?;

    let mut host_b = vec![T::ZERO; b_op.buffer.len()];
    device.download(b_op.buffer, &mut host_b)?;

    let view_b = leto::ArrayView2::new(*b_op.layout, &host_b);

    let mut host_c = vec![T::ZERO; expected_c_len];
    leto_ops::spmm_into(&cpu_a, &view_b, &mut host_c).map_err(|e| {
        HephaestusError::DispatchFailed {
            message: format!("spmm_into failed: {e}"),
        }
    })?;

    device.write_buffer(c, &host_c)
}

/// Compute `C = A · B`, allocating the result buffer.
pub fn spmm<'a, T: CudaScalar + leto_ops::Scalar + Pod, B: AsGpuMatrixOperand<'a, T>>(
    device: &CudaDevice,
    a: &GpuCsrMatrix<T>,
    b: &B,
) -> Result<CudaBuffer<T>> {
    let (nrows, _) = a.shape();
    let b_op = b.as_operand();
    let [_, bcols] = b_op.layout.shape;

    let mut c = device.alloc_zeroed::<T>(nrows * bcols)?;
    spmm_into(device, a, b, &mut c)?;
    Ok(c)
}
