#![cfg(feature = "cuda")]

//! Row-major matrix-region transfers for CUDA backend.
//!
//! The implementation uses row-wise 1D copies rather than `CUDA_MEMCPY2D`.
//! cuda-oxide 0.4.0 generates `size_t` as `c_ulong`; on Windows/MSVC that
//! makes the `CUDA_MEMCPY2D` layout incompatible with the CUDA driver ABI.

use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::{cuda_byte_count, CudaDevice};
use hephaestus_core::{HephaestusError, Result};

#[derive(Clone, Copy)]
pub(crate) struct MatrixRegion {
    pub(crate) stride: usize,
    pub(crate) row_start: usize,
    pub(crate) col_start: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub(crate) fn download_matrix_region_compact(
    device: &CudaDevice,
    buffer: &CudaBuffer<f32>,
    region: MatrixRegion,
) -> Result<Vec<f32>> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(vec![]);
    }
    device.bind()?;
    let size_of_f32 = std::mem::size_of::<f32>();
    let row_bytes = region.cols * size_of_f32;
    let row_byte_count = cuda_byte_count(row_bytes, "matrix region row byte count")?;

    let mut compact = vec![0.0f32; region.rows * region.cols];

    let device_start_index = region
        .row_start
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region device offset overflows usize".to_string(),
        })?;
    let device_needed_len = device_start_index + (region.rows - 1) * region.stride + region.cols;
    if device_needed_len > buffer.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: device_needed_len,
            device_len: buffer.len,
        });
    }

    for row in 0..region.rows {
        let source_index = device_start_index + row * region.stride;
        let source_offset = source_index
            .checked_mul(size_of_f32)
            .and_then(|b| u64::try_from(b).ok())
            .ok_or_else(|| HephaestusError::TransferFailed {
                message: "matrix region row byte offset overflows u64".to_string(),
            })?;
        let dest = compact.as_mut_ptr().wrapping_add(row * region.cols);
        // SAFETY: this device's context is current (`bind` above). Each row
        // reads `cols` contiguous f32 values from a bounds-checked device row
        // and writes the matching row in the live compact Vec.
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyDtoH_v2(
                dest.cast::<std::ffi::c_void>(),
                buffer.raw() + source_offset,
                row_byte_count,
            )
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "download_matrix_region_compact row {row} cuMemcpyDtoH_v2 failed: {res}"
                ),
            });
        }
    }
    Ok(compact)
}

pub(crate) fn write_matrix_region_compact(
    device: &CudaDevice,
    buffer: &CudaBuffer<f32>,
    compact_host: &[f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }
    let compact_len = region.rows * region.cols;
    if compact_host.len() != compact_len {
        return Err(HephaestusError::TransferFailed {
            message: format!(
                "write_matrix_region_compact length mismatch: compact_host len {}, expected {}",
                compact_host.len(),
                compact_len
            ),
        });
    }
    device.bind()?;
    let size_of_f32 = std::mem::size_of::<f32>();
    let row_bytes = region.cols * size_of_f32;
    let row_byte_count = cuda_byte_count(row_bytes, "matrix region row byte count")?;

    let device_start_index = region
        .row_start
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region device offset overflows usize".to_string(),
        })?;
    let device_needed_len = device_start_index + (region.rows - 1) * region.stride + region.cols;
    if device_needed_len > buffer.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: device_needed_len,
            device_len: buffer.len,
        });
    }

    for row in 0..region.rows {
        let dest_index = device_start_index + row * region.stride;
        let dest_offset = dest_index
            .checked_mul(size_of_f32)
            .and_then(|b| u64::try_from(b).ok())
            .ok_or_else(|| HephaestusError::TransferFailed {
                message: "matrix region row byte offset overflows u64".to_string(),
            })?;
        let source = compact_host.as_ptr().wrapping_add(row * region.cols);
        // SAFETY: this device's context is current (`bind` above). Each row
        // reads `cols` contiguous f32 values from the length-checked compact
        // host slice and writes the matching bounds-checked device row.
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyHtoD_v2(
                buffer.raw() + dest_offset,
                source.cast::<std::ffi::c_void>(),
                row_byte_count,
            )
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "write_matrix_region_compact row {row} cuMemcpyHtoD_v2 failed: {res}"
                ),
            });
        }
    }
    Ok(())
}
