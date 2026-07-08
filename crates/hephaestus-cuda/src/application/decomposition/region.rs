#![cfg(feature = "cuda")]

//! Row-major matrix-region transfers for CUDA backend.
//!
//! The implementation uses row-wise 1D copies rather than `CUDA_MEMCPY2D`.
//! cuda-oxide 0.4.0 generates `size_t` as `c_ulong`; on Windows/MSVC that
//! makes the `CUDA_MEMCPY2D` layout incompatible with the CUDA driver ABI.
//!
//! Both directions stage through [`PinnedHostBuffer`] rather than a plain
//! `Vec<f32>` and issue the async copy variants per row, syncing once after
//! all rows are enqueued (CU-P6/CU-M3): pinned memory gets full-bandwidth DMA
//! transfers instead of the driver's implicit pageable-memory staging copy,
//! and enqueuing every row before the single sync lets the driver pipeline
//! the per-row copies instead of serializing a blocking call per row.

use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::{cuda_byte_count, CudaDevice};
use crate::infrastructure::pinned::PinnedHostBuffer;
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
) -> Result<PinnedHostBuffer<f32>> {
    if region.rows == 0 || region.cols == 0 {
        return PinnedHostBuffer::zeroed(device.cuda_context().clone(), 0);
    }
    device.bind()?;
    let size_of_f32 = std::mem::size_of::<f32>();
    let row_bytes = region.cols * size_of_f32;
    let row_byte_count = cuda_byte_count(row_bytes, "matrix region row byte count")?;

    let mut compact =
        PinnedHostBuffer::<f32>::zeroed(device.cuda_context().clone(), region.rows * region.cols)?;

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
        let dest = compact
            .as_mut_ptr()
            .cast::<f32>()
            .wrapping_add(row * region.cols);
        // SAFETY: this device's context is current (`bind` above). Each row
        // reads `cols` contiguous f32 values from a bounds-checked device row
        // into the matching row of `compact`'s pinned allocation, which
        // outlives this loop (owned by `compact`, not freed until its own
        // `Drop`). The copy is enqueued asynchronously on the legacy stream;
        // `cuStreamSynchronize` below waits for every enqueued row before any
        // host read of `compact`.
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyDtoHAsync_v2(
                dest.cast::<std::ffi::c_void>(),
                buffer.raw() + source_offset,
                row_byte_count,
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "download_matrix_region_compact row {row} cuMemcpyDtoHAsync_v2 failed: {res}"
                ),
            });
        }
    }
    // SAFETY: context is current (`bind` above); waits on the legacy/null
    // stream every enqueued row above was issued on, making the transfers
    // host-visible before this function returns `compact` for host reads.
    let sync_res = unsafe { cuda_oxide::sys::cuStreamSynchronize(std::ptr::null_mut()) };
    if sync_res != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!(
                "download_matrix_region_compact cuStreamSynchronize failed: {sync_res}"
            ),
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
        // host slice into the matching bounds-checked device row. The copy is
        // enqueued asynchronously on the legacy stream; `cuStreamSynchronize`
        // below waits for every enqueued row before this function returns, so
        // `compact_host` (borrowed only for this call) stays valid for the
        // driver's read regardless of whether the caller's backing storage
        // happens to be pinned.
        let res = unsafe {
            cuda_oxide::sys::cuMemcpyHtoDAsync_v2(
                buffer.raw() + dest_offset,
                source.cast::<std::ffi::c_void>(),
                row_byte_count,
                std::ptr::null_mut(),
            )
        };
        if res != 0 {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "write_matrix_region_compact row {row} cuMemcpyHtoDAsync_v2 failed: {res}"
                ),
            });
        }
    }
    // SAFETY: context is current (`bind` above); waits on the legacy/null
    // stream every enqueued row above was issued on, making the writes
    // complete (and `compact_host` safe to drop/reuse) before returning.
    let sync_res = unsafe { cuda_oxide::sys::cuStreamSynchronize(std::ptr::null_mut()) };
    if sync_res != 0 {
        return Err(HephaestusError::TransferFailed {
            message: format!("write_matrix_region_compact cuStreamSynchronize failed: {sync_res}"),
        });
    }
    Ok(())
}
