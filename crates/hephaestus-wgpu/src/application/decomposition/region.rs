//! Row-major matrix-region transfers for hybrid decomposition kernels.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use std::any::TypeId;

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::UniformBufferGuard;

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable)]
struct RegionCopyMeta {
    stride: u32,
    row_start: u32,
    col_start: u32,
    rows: u32,
    cols: u32,
}

// SAFETY: RegionCopyMeta is `#[repr(C)]` and every field is Pod.
unsafe impl bytemuck::Pod for RegionCopyMeta {}

struct RegionCopyKernel;

fn region_gather_shader_source() -> String {
    r#"struct RegionCopyMeta {
    stride: u32,
    row_start: u32,
    col_start: u32,
    rows: u32,
    cols: u32,
}
@group(0) @binding(0) var<storage, read> src_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst_compact: array<f32>;
@group(0) @binding(2) var<uniform> params: RegionCopyMeta;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let total_elements = params.rows * params.cols;
    let idx = gid.x;
    if (idx >= total_elements) {
        return;
    }
    let r = idx / params.cols;
    let c = idx % params.cols;
    let src_idx = (params.row_start + r) * params.stride + (params.col_start + c);
    dst_compact[idx] = src_matrix[src_idx];
}
"#
    .to_string()
}

fn region_scatter_shader_source() -> String {
    r#"struct RegionCopyMeta {
    stride: u32,
    row_start: u32,
    col_start: u32,
    rows: u32,
    cols: u32,
}
@group(0) @binding(0) var<storage, read_write> dst_matrix: array<f32>;
@group(0) @binding(1) var<storage, read> src_compact: array<f32>;
@group(0) @binding(2) var<uniform> params: RegionCopyMeta;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let total_elements = params.rows * params.cols;
    let idx = gid.x;
    if (idx >= total_elements) {
        return;
    }
    let r = idx / params.cols;
    let c = idx % params.cols;
    let dst_idx = (params.row_start + r) * params.stride + (params.col_start + c);
    dst_matrix[dst_idx] = src_compact[idx];
}
"#
    .to_string()
}

/// Convert a `usize` workgroup count to `u32`, returning `DispatchFailed` on overflow.
fn checked_wg_x(wg_x: usize) -> Result<u32> {
    u32::try_from(wg_x).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("region kernel workgroup count {wg_x} exceeds u32::MAX"),
    })
}

/// Convert a region field to `u32`, returning `TransferFailed` on overflow.
fn region_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::TransferFailed {
        message: format!("region {name} {value} exceeds u32"),
    })
}

/// Build a `RegionCopyMeta` from a `MatrixRegion`, checking all field widths.
fn region_meta(region: MatrixRegion) -> Result<RegionCopyMeta> {
    Ok(RegionCopyMeta {
        stride: region_u32(region.stride, "stride")?,
        row_start: region_u32(region.row_start, "row_start")?,
        col_start: region_u32(region.col_start, "col_start")?,
        rows: region_u32(region.rows, "rows")?,
        cols: region_u32(region.cols, "cols")?,
    })
}

// ---------------------------------------------------------------------------
// Core reusable implementation — callers supply the compact device buffer
// ---------------------------------------------------------------------------

/// Gather a matrix region from `buffer` into a freshly-allocated host `Vec<f32>`.
#[allow(dead_code)]
pub(crate) fn download_matrix_region_compact(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    region: MatrixRegion,
) -> Result<Vec<f32>> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(vec![]);
    }
    let compact_len = matrix_region_len(region.rows, region.cols)?;
    let temp = device.alloc_zeroed::<f32>(compact_len)?;
    download_matrix_region_compact_reusable(device, buffer, &temp, region)
}

/// Scatter `compact_host` into a region of `buffer` using a fresh temporary
/// device buffer.
#[allow(dead_code)]
pub(crate) fn write_matrix_region_compact(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    compact_host: &[f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }
    let compact_len = matrix_region_len(region.rows, region.cols)?;
    let temp = device.alloc_zeroed::<f32>(compact_len)?;
    write_matrix_region_compact_reusable(device, buffer, &temp, compact_host, region)
}

/// Gather a matrix region from `buffer` into caller-supplied `temp_compact_buf`
/// and return the region as a host `Vec<f32>`.
///
/// `temp_compact_buf` must hold at least `region.rows * region.cols` elements.
pub(crate) fn download_matrix_region_compact_reusable(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    temp_compact_buf: &WgpuBuffer<f32>,
    region: MatrixRegion,
) -> Result<Vec<f32>> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(vec![]);
    }

    let compact_len = matrix_region_len(region.rows, region.cols)?;
    let compact_bytes = WgpuDevice::byte_size::<f32>(compact_len)?;

    if temp_compact_buf.len < compact_len {
        return Err(HephaestusError::TransferFailed {
            message: format!(
                "reusable temp_compact_buf has insufficient capacity: {}, expected at least {}",
                temp_compact_buf.len, compact_len
            ),
        });
    }

    let raw_staging = device.get_staging_buffer(compact_bytes)?;
    let staging_size = raw_staging.size();
    let staging = crate::infrastructure::pool::StagingBufferGuard::new(device.clone(), raw_staging);

    let raw_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<RegionCopyMeta>(1)?)?;
    let meta_buf = UniformBufferGuard::new(device.clone(), raw_meta_buf);

    let meta = region_meta(region)?;
    device
        .queue()
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<RegionCopyKernel>(), TypeId::of::<f32>(), 0),
        "hephaestus-region-gather",
        region_gather_shader_source,
    );

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-region-gather-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_compact_buf.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matrix-region-download-compact"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-region-gather-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = checked_wg_x(compact_len.div_ceil(256))?;
        pass.dispatch_workgroups(wg_x, 1, 1);
    }

    encoder.copy_buffer_to_buffer(temp_compact_buf.raw(), 0, &staging, 0, compact_bytes);

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
    let mut compact = vec![0.0f32; compact_len];
    compact.copy_from_slice(bytemuck::cast_slice(&mapped[..compact_bytes as usize]));
    drop(mapped);
    staging.unmap();

    Ok(compact)
}

/// Scatter `compact_host` into a region of `buffer` via caller-supplied
/// `temp_compact_buf`.
///
/// `temp_compact_buf` must hold at least `region.rows * region.cols` elements.
pub(crate) fn write_matrix_region_compact_reusable(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    temp_compact_buf: &WgpuBuffer<f32>,
    compact_host: &[f32],
    region: MatrixRegion,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        return Ok(());
    }

    let compact_len = matrix_region_len(region.rows, region.cols)?;
    if compact_host.len() != compact_len {
        return Err(HephaestusError::TransferFailed {
            message: format!(
                "write_matrix_region_compact length mismatch: compact_host len {}, expected {}",
                compact_host.len(),
                compact_len
            ),
        });
    }

    if temp_compact_buf.len < compact_len {
        return Err(HephaestusError::TransferFailed {
            message: format!(
                "reusable temp_compact_buf has insufficient capacity: {}, expected at least {}",
                temp_compact_buf.len, compact_len
            ),
        });
    }

    device.write_sub_buffer(temp_compact_buf, 0, compact_host)?;

    let raw_meta_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<RegionCopyMeta>(1)?)?;
    let meta_buf = UniformBufferGuard::new(device.clone(), raw_meta_buf);

    let meta = region_meta(region)?;
    device
        .queue()
        .write_buffer(&meta_buf, 0, bytemuck::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<RegionCopyKernel>(), TypeId::of::<f32>(), 1),
        "hephaestus-region-scatter",
        region_scatter_shader_source,
    );

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-region-scatter-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_compact_buf.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matrix-region-upload-compact"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-region-scatter-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = checked_wg_x(compact_len.div_ceil(256))?;
        pass.dispatch_workgroups(wg_x, 1, 1);
    }

    device.queue().submit(Some(encoder.finish()));

    Ok(())
}

