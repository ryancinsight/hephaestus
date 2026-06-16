//! Pseudoinverse and matrix exponential on CUDA device.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Compute the Moore-Penrose pseudoinverse A⁺ on the GPU.
pub fn pinv(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<CudaBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 || cols == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let out_arr = leto_ops::pinv(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Pseudoinverse failed: {e}"),
    })?;

    device.upload(leto::Storage::as_slice(out_arr.storage()))
}

/// Compute the matrix exponential e^A on the GPU.
pub fn matexp(
    device: &CudaDevice,
    matrix: StridedOperand<'_, f32, 2>,
) -> Result<CudaBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Matrix exponential requires square matrix, got shape [{rows}, {cols}]"),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;

    if rows == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;

    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let out_arr = leto_ops::matexp(&view).map_err(|e| HephaestusError::DispatchFailed {
        message: format!("Matrix exponential failed: {e}"),
    })?;

    device.upload(leto::Storage::as_slice(out_arr.storage()))
}
