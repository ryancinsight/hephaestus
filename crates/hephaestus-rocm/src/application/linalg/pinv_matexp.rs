//! ROCm pseudoinverse and matrix exponential through the shared provider.

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError, Result};

use super::map_layout_err;
use crate::RocmDevice;
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

/// Compute the Moore–Penrose pseudoinverse **A⁺** through the shared provider.
pub fn pinv(device: &RocmDevice, matrix: StridedOperand<'_, f32, 2>) -> Result<RocmBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape;
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    if rows == 0 || cols == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let output = leto_ops::pinv(&view).map_err(|error| HephaestusError::DispatchFailed {
        message: format!("Pseudoinverse failed: {error}"),
    })?;
    let output_slice = leto::Storage::as_slice(output.storage());
    device.upload(output_slice)
}

/// Compute the matrix exponential **eᴬ** through the shared provider.
pub fn matexp(device: &RocmDevice, matrix: StridedOperand<'_, f32, 2>) -> Result<RocmBuffer<f32>> {
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Matrix exponential requires square matrix, got shape [{rows}, {cols}]"
            ),
        });
    }
    matrix
        .layout
        .validate_storage_len(matrix.buffer.len())
        .map_err(map_layout_err)?;
    if rows == 0 {
        return device.alloc_zeroed::<f32>(0);
    }

    let mut host_data = vec![0.0_f32; matrix.buffer.len()];
    device.download(matrix.buffer, &mut host_data)?;
    let view = leto::ArrayView::<f32, 2>::new(*matrix.layout, &host_data);
    let output = leto_ops::matexp(&view).map_err(|error| HephaestusError::DispatchFailed {
        message: format!("Matrix exponential failed: {error}"),
    })?;
    let output_slice = leto::Storage::as_slice(output.storage());
    device.upload(output_slice)
}
