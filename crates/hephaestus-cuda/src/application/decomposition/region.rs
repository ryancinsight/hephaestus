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

pub(crate) fn write_matrix_region(
    device: &CudaDevice,
    buffer: &CudaBuffer<f32>,
    host: &[f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }
    device.bind()?;
    let size_of_f32 = std::mem::size_of::<f32>();
    let row_bytes = region.cols * size_of_f32;

    let host_start_index = region.row_start
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region host offset overflows usize".to_string(),
        })?;

    let host_last_row_start = (region.row_start + region.rows - 1)
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region last row offset overflows usize".to_string(),
        })?;
    let host_needed_len = host_last_row_start.checked_add(region.cols).ok_or_else(|| {
        HephaestusError::TransferFailed {
            message: "matrix region host needed length overflows usize".to_string(),
        }
    })?;
    if host.len() < host_needed_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: host_needed_len,
            device_len: host.len(),
        });
    }

    let host_ptr = unsafe { host.as_ptr().add(host_start_index) };
    let device_offset = (host_start_index * size_of_f32) as u64;

    let device_needed_len = host_start_index + (region.rows - 1) * region.stride + region.cols;
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
        srcHost: host_ptr as *const std::ffi::c_void,
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: region.stride * size_of_f32,

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

    let res = unsafe { cuda_core::sys::cuMemcpy2D_v2(&mut copy as *mut cuda_core::sys::CUDA_MEMCPY2D) };
    if res != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!("write_matrix_region cuMemcpy2D_v2 failed: {res}"),
        });
    }
    Ok(())
}

pub(crate) fn download_matrix_region(
    device: &CudaDevice,
    buffer: &CudaBuffer<f32>,
    host: &mut [f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }
    device.bind()?;
    let size_of_f32 = std::mem::size_of::<f32>();
    let row_bytes = region.cols * size_of_f32;

    let host_start_index = region.row_start
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region host offset overflows usize".to_string(),
        })?;

    let host_last_row_start = (region.row_start + region.rows - 1)
        .checked_mul(region.stride)
        .and_then(|base| base.checked_add(region.col_start))
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: "matrix region last row offset overflows usize".to_string(),
        })?;
    let host_needed_len = host_last_row_start.checked_add(region.cols).ok_or_else(|| {
        HephaestusError::TransferFailed {
            message: "matrix region host needed length overflows usize".to_string(),
        }
    })?;
    if host.len() < host_needed_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: host_needed_len,
            device_len: host.len(),
        });
    }

    let host_ptr = unsafe { host.as_mut_ptr().add(host_start_index) };
    let device_offset = (host_start_index * size_of_f32) as u64;

    let device_needed_len = host_start_index + (region.rows - 1) * region.stride + region.cols;
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
        dstHost: host_ptr as *mut std::ffi::c_void,
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: region.stride * size_of_f32,

        WidthInBytes: row_bytes,
        Height: region.rows,
    };

    let res = unsafe { cuda_core::sys::cuMemcpy2D_v2(&mut copy as *mut cuda_core::sys::CUDA_MEMCPY2D) };
    if res != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!("download_matrix_region cuMemcpy2D_v2 failed: {res}"),
        });
    }
    Ok(())
}
