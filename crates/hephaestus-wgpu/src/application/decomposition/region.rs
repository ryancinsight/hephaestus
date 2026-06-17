//! Row-major matrix-region transfers for hybrid decomposition kernels.

use hephaestus_core::{HephaestusError, Result};

use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

fn element_byte_offset(index: usize) -> Result<u64> {
    index
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: format!("element offset {index} overflows byte offset"),
        })
}

fn matrix_region_len(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| HephaestusError::TransferFailed {
            message: format!("matrix region shape [{rows}, {cols}] overflows element count"),
        })
}

#[derive(Clone, Copy)]
pub(crate) struct MatrixRegion {
    pub(crate) stride: usize,
    pub(crate) row_start: usize,
    pub(crate) col_start: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub(crate) fn write_matrix_region(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    host: &[f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }
    let row_bytes = WgpuDevice::byte_size::<f32>(region.cols)?;
    for row in 0..region.rows {
        let host_offset = (region.row_start + row)
            .checked_mul(region.stride)
            .and_then(|base| base.checked_add(region.col_start))
            .ok_or_else(|| HephaestusError::TransferFailed {
                message: "matrix region host offset overflows usize".to_string(),
            })?;
        let end = host_offset.checked_add(region.cols).ok_or_else(|| {
            HephaestusError::TransferFailed {
                message: "matrix region host end overflows usize".to_string(),
            }
        })?;
        let device_offset = element_byte_offset(host_offset)?;
        device.queue().write_buffer(
            buffer.raw(),
            device_offset,
            bytemuck::cast_slice(&host[host_offset..end]),
        );
        debug_assert_eq!(row_bytes as usize, region.cols * std::mem::size_of::<f32>());
    }
    Ok(())
}

pub(crate) fn download_matrix_region(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    host: &mut [f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }

    let compact_len = matrix_region_len(region.rows, region.cols)?;
    let compact_bytes = WgpuDevice::byte_size::<f32>(compact_len)?;
    let row_bytes = WgpuDevice::byte_size::<f32>(region.cols)?;
    let staging = device.get_staging_buffer(compact_bytes)?;
    let staging_size = staging.size();

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matrix-region-download"),
        });
    for row in 0..region.rows {
        let source_index = (region.row_start + row)
            .checked_mul(region.stride)
            .and_then(|base| base.checked_add(region.col_start))
            .ok_or_else(|| HephaestusError::TransferFailed {
                message: "matrix region source offset overflows usize".to_string(),
            })?;
        let source_offset = element_byte_offset(source_index)?;
        let dest_offset = WgpuDevice::byte_size::<f32>(row * region.cols)?;
        encoder.copy_buffer_to_buffer(
            buffer.raw(),
            source_offset,
            &staging,
            dest_offset,
            row_bytes,
        );
    }
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..staging_size);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .inner()
        .poll(wgpu::PollType::Wait)
        .map_err(|e| HephaestusError::TransferFailed {
            message: format!("device poll failed: {e:?}"),
        })?;
    receiver
        .recv()
        .map_err(|_| HephaestusError::TransferFailed {
            message: "map_async callback dropped".to_string(),
        })?
        .map_err(|e| HephaestusError::TransferFailed {
            message: format!("buffer mapping failed: {e:?}"),
        })?;

    let mapped = slice.get_mapped_range();
    let compact: &[f32] = bytemuck::cast_slice(&mapped[..compact_bytes as usize]);
    for row in 0..region.rows {
        let host_offset = (region.row_start + row)
            .checked_mul(region.stride)
            .and_then(|base| base.checked_add(region.col_start))
            .ok_or_else(|| HephaestusError::TransferFailed {
                message: "matrix region host offset overflows usize".to_string(),
            })?;
        let host_end = host_offset.checked_add(region.cols).ok_or_else(|| {
            HephaestusError::TransferFailed {
                message: "matrix region host end overflows usize".to_string(),
            }
        })?;
        let compact_offset = row * region.cols;
        host[host_offset..host_end]
            .copy_from_slice(&compact[compact_offset..compact_offset + region.cols]);
    }
    drop(mapped);
    staging.unmap();

    device.recycle_staging_buffer(staging);
    Ok(())
}
