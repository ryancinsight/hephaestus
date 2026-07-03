#![cfg(feature = "cuda")]

//! Row-major matrix-region transfers for CUDA backend.

use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;
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

    let mut compact = vec![0.0f32; region.rows * region.cols];

    let device_start_index = region
        .row_start
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region device offset overflows usize".to_string(),
        })?;
    let device_offset = device_start_index
        .checked_mul(size_of_f32)
        .and_then(|b| u64::try_from(b).ok())
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region device byte offset overflows u64".to_string(),
        })?;

    let device_needed_len = device_start_index + (region.rows - 1) * region.stride + region.cols;
    if device_needed_len > buffer.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: device_needed_len,
            device_len: buffer.len,
        });
    }

    let mut copy = cuda_core::sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: buffer.raw() + device_offset,
        srcArray: std::ptr::null_mut(),
        srcPitch: region.stride * size_of_f32,

        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_HOST,
        dstHost: compact.as_mut_ptr() as *mut std::ffi::c_void,
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: region.cols * size_of_f32,

        WidthInBytes: row_bytes,
        Height: region.rows,
    };

    // SAFETY: this device's context is current (`bind` above). 2D copy
    // geometry: the source reads `rows` rows of `WidthInBytes = cols * 4`
    // at `srcPitch = stride * 4`; the farthest device element touched,
    // `device_start_index + (rows - 1) * stride + cols`, is bounds-checked
    // against `buffer.len` above (every earlier row ends no later, strides
    // are unsigned). The destination covers exactly `rows * cols` elements
    // of the live `compact` Vec (`dstPitch == WidthInBytes`, `Height ==
    // rows`). The driver rejects `pitch < WidthInBytes` with an error code
    // surfaced below, and a pageable-host-memory `cuMemcpy2D_v2` is
    // synchronous, so `compact` is not referenced after this call returns.
    let res =
        unsafe { cuda_core::sys::cuMemcpy2D_v2(&mut copy as *mut cuda_core::sys::CUDA_MEMCPY2D) };
    if res != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!("download_matrix_region_compact cuMemcpy2D_v2 failed: {res}"),
        });
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

    let device_start_index = region
        .row_start
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region device offset overflows usize".to_string(),
        })?;
    let device_offset = device_start_index
        .checked_mul(size_of_f32)
        .and_then(|b| u64::try_from(b).ok())
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region device byte offset overflows u64".to_string(),
        })?;

    let device_needed_len = device_start_index + (region.rows - 1) * region.stride + region.cols;
    if device_needed_len > buffer.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: device_needed_len,
            device_len: buffer.len,
        });
    }

    let mut copy = cuda_core::sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_HOST,
        srcHost: compact_host.as_ptr() as *const std::ffi::c_void,
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: region.cols * size_of_f32,

        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: buffer.raw() + device_offset,
        dstArray: std::ptr::null_mut(),
        dstPitch: region.stride * size_of_f32,

        WidthInBytes: row_bytes,
        Height: region.rows,
    };

    // SAFETY: this device's context is current (`bind` above). 2D copy
    // geometry: the source reads exactly `rows * cols` elements of the live
    // `compact_host` slice (length validated above, `srcPitch ==
    // WidthInBytes = cols * 4`, `Height == rows`); the farthest device
    // element written, `device_start_index + (rows - 1) * stride + cols`, is
    // bounds-checked against `buffer.len` above (every earlier row ends no
    // later, strides are unsigned). The driver rejects `pitch <
    // WidthInBytes` with an error code surfaced below, and a
    // pageable-host-memory `cuMemcpy2D_v2` is synchronous, so `compact_host`
    // is not referenced after this call returns.
    let res =
        unsafe { cuda_core::sys::cuMemcpy2D_v2(&mut copy as *mut cuda_core::sys::CUDA_MEMCPY2D) };
    if res != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!("write_matrix_region_compact cuMemcpy2D_v2 failed: {res}"),
        });
    }
    Ok(())
}
