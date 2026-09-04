//! Row-major matrix-region transfers for hybrid decomposition kernels.

use hephaestus_core::{HephaestusError, Result};
use std::any::TypeId;

use crate::application::pipeline::cached_pipeline;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::{
    StagingBufferGuard, UniformBufferGuard, staging_guard, uniform_guard,
};

pub(crate) fn matrix_region_len(rows: usize, cols: usize) -> Result<usize> {
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

pub(crate) struct MatrixRegionUpload<'a> {
    pub(crate) temp: &'a WgpuBuffer<f32>,
    pub(crate) host: &'a [f32],
    pub(crate) region: MatrixRegion,
}

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct RegionCopyMeta {
    stride: u32,
    row_start: u32,
    col_start: u32,
    rows: u32,
    cols: u32,
}

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

pub(crate) struct MatrixRegionDownloadWorkspace<'buffer> {
    device: WgpuDevice,
    _source: &'buffer WgpuBuffer<f32>,
    temp: &'buffer WgpuBuffer<f32>,
    pipeline: wgpu::ComputePipeline,
    staging: StagingBufferGuard,
    staging_size: u64,
    _meta: UniformBufferGuard,
    bind_group: wgpu::BindGroup,
    capacity: usize,
}

struct GatherTransfer<'workspace, 'buffer> {
    workspace: &'workspace MatrixRegionDownloadWorkspace<'buffer>,
    compact_len: usize,
    compact_bytes: u64,
}

type MappingResult = core::result::Result<(), wgpu::BufferAsyncError>;

fn send_mapping_result(sender: std::sync::mpsc::Sender<MappingResult>, result: MappingResult) {
    let Err(orphaned_result) = sender.send(result) else {
        return;
    };
    // The synchronous caller owns the only receiver. If it has already
    // unwound after a device-poll failure, no observer remains for this value.
    drop(orphaned_result.0);
}

impl<'buffer> MatrixRegionDownloadWorkspace<'buffer> {
    pub(crate) fn new(
        device: &WgpuDevice,
        source: &'buffer WgpuBuffer<f32>,
        temp: &'buffer WgpuBuffer<f32>,
        capacity: usize,
    ) -> Result<Self> {
        if temp.len < capacity {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "reusable temp buffer has insufficient capacity: {}, expected at least {}",
                    temp.len, capacity
                ),
            });
        }
        let staging_bytes = WgpuDevice::byte_size::<f32>(capacity)?;
        let raw_staging = device.get_staging_buffer(staging_bytes)?;
        let staging_size = raw_staging.size();
        let staging = staging_guard(device.clone(), raw_staging);
        let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<RegionCopyMeta>(1)?)?;
        let meta = uniform_guard(device.clone(), raw_meta);
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
                        resource: source.raw().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: temp.raw().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: meta.as_entire_binding(),
                    },
                ],
            });
        Ok(Self {
            device: device.clone(),
            _source: source,
            temp,
            pipeline,
            staging,
            staging_size,
            _meta: meta,
            bind_group,
            capacity,
        })
    }

    fn prepare<'workspace>(
        &'workspace self,
        region: MatrixRegion,
    ) -> Result<GatherTransfer<'workspace, 'buffer>> {
        let compact_len = matrix_region_len(region.rows, region.cols)?;
        if compact_len > self.capacity {
            return Err(HephaestusError::TransferFailed {
                message: format!(
                    "download workspace has insufficient capacity: {}, expected at least {}",
                    self.capacity, compact_len
                ),
            });
        }
        self.device.queue().write_buffer(
            &self._meta,
            0,
            eunomia::layout::bytes_of(&region_meta(region)?),
        );
        Ok(GatherTransfer {
            workspace: self,
            compact_len,
            compact_bytes: WgpuDevice::byte_size::<f32>(compact_len)?,
        })
    }

    pub(crate) fn download_into(&self, region: MatrixRegion, out: &mut Vec<f32>) -> Result<()> {
        if region.rows == 0 || region.cols == 0 {
            out.clear();
            return Ok(());
        }
        let transfer = self.prepare(region)?;
        let mut encoder =
            self.device
                .inner()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("hephaestus-matrix-region-download-compact"),
                });
        encode_gather(&mut encoder, &transfer)?;
        self.device.queue().submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        self.staging
            .slice(..self.staging_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                send_mapping_result(sender, result);
            });
        if let Err(error) = wait_for_mappings(&self.device, [receiver]) {
            self.staging.unmap();
            return Err(error);
        }
        copy_mapped_gather(&transfer, out)
    }
}

fn encode_gather(
    encoder: &mut wgpu::CommandEncoder,
    transfer: &GatherTransfer<'_, '_>,
) -> Result<()> {
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-region-gather-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&transfer.workspace.pipeline);
        pass.set_bind_group(0, &transfer.workspace.bind_group, &[]);
        pass.dispatch_workgroups(checked_wg_x(transfer.compact_len.div_ceil(256))?, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        transfer.workspace.temp.raw(),
        0,
        &transfer.workspace.staging,
        0,
        transfer.compact_bytes,
    );
    Ok(())
}

fn copy_mapped_gather(transfer: &GatherTransfer<'_, '_>, out: &mut Vec<f32>) -> Result<()> {
    let mapped = match transfer
        .workspace
        .staging
        .slice(..transfer.workspace.staging_size)
        .get_mapped_range()
    {
        Ok(mapped) => mapped,
        Err(error) => {
            transfer.workspace.staging.unmap();
            return Err(HephaestusError::TransferFailed {
                message: format!("mapped-range acquisition failed: {error}"),
            });
        }
    };
    out.resize(transfer.compact_len, 0.0);
    let compact_bytes = transfer
        .compact_len
        .checked_mul(core::mem::size_of::<f32>())
        .expect("invariant: compact byte size was validated by WgpuDevice::byte_size");
    out.copy_from_slice(eunomia::layout::cast_slice(&mapped[..compact_bytes]));
    drop(mapped);
    transfer.workspace.staging.unmap();
    Ok(())
}

fn wait_for_mappings(
    device: &WgpuDevice,
    receivers: impl IntoIterator<
        Item = std::sync::mpsc::Receiver<core::result::Result<(), wgpu::BufferAsyncError>>,
    >,
) -> Result<()> {
    let deadline = crate::infrastructure::device::device_wait_deadline();
    device
        .inner()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(deadline),
        })
        .map_err(|error| {
            crate::infrastructure::device::poll_failure(
                "decomposition region mapping wait",
                deadline,
                &error,
            )
        })?;
    for receiver in receivers {
        receiver
            .recv()
            .map_err(|_| HephaestusError::TransferFailed {
                message: "map_async callback dropped".to_string(),
            })?
            .map_err(|error| HephaestusError::TransferFailed {
                message: format!("buffer mapping failed: {error:?}"),
            })?;
    }
    Ok(())
}

/// Gather a matrix region from `buffer` into caller-supplied `temp_compact_buf`
/// and write the result into `out`, resized to `region.rows * region.cols`.
///
/// `out`'s existing allocation is reused (`resize` keeps capacity), so a caller
/// looping over panels allocates the host buffer once and refills it each
/// iteration rather than allocating a fresh `Vec` per call. This is the SSOT for
/// region downloads.
///
/// `temp_compact_buf` must hold at least `region.rows * region.cols` elements.
pub(crate) fn download_matrix_region_compact_into(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    temp_compact_buf: &WgpuBuffer<f32>,
    region: MatrixRegion,
    out: &mut Vec<f32>,
) -> Result<()> {
    if region.rows == 0 || region.cols == 0 {
        out.clear();
        return Ok(());
    }
    let capacity = matrix_region_len(region.rows, region.cols)?;
    MatrixRegionDownloadWorkspace::new(device, buffer, temp_compact_buf, capacity)?
        .download_into(region, out)
}

pub(crate) fn download_matrix_region_workspace_pair_into(
    first_workspace: &MatrixRegionDownloadWorkspace<'_>,
    first_region: MatrixRegion,
    first_out: &mut Vec<f32>,
    second_workspace: &MatrixRegionDownloadWorkspace<'_>,
    second_region: MatrixRegion,
    second_out: &mut Vec<f32>,
) -> Result<()> {
    if !std::sync::Arc::ptr_eq(
        first_workspace.device.device(),
        second_workspace.device.device(),
    ) {
        return Err(HephaestusError::TransferFailed {
            message: "paired download workspaces belong to different devices".to_string(),
        });
    }
    if first_region.rows == 0 || first_region.cols == 0 {
        first_out.clear();
        return second_workspace.download_into(second_region, second_out);
    }
    if second_region.rows == 0 || second_region.cols == 0 {
        second_out.clear();
        return first_workspace.download_into(first_region, first_out);
    }

    let first_transfer = first_workspace.prepare(first_region)?;
    let second_transfer = second_workspace.prepare(second_region)?;
    let device = &first_workspace.device;
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matrix-region-pair-download-compact"),
        });
    encode_gather(&mut encoder, &first_transfer)?;
    encode_gather(&mut encoder, &second_transfer)?;
    device.queue().submit(Some(encoder.finish()));

    let (first_sender, first_receiver) = std::sync::mpsc::channel();
    first_workspace
        .staging
        .slice(..first_workspace.staging_size)
        .map_async(wgpu::MapMode::Read, move |result| {
            send_mapping_result(first_sender, result);
        });
    let (second_sender, second_receiver) = std::sync::mpsc::channel();
    second_workspace
        .staging
        .slice(..second_workspace.staging_size)
        .map_async(wgpu::MapMode::Read, move |result| {
            send_mapping_result(second_sender, result);
        });
    if let Err(error) = wait_for_mappings(device, [first_receiver, second_receiver]) {
        first_workspace.staging.unmap();
        second_workspace.staging.unmap();
        return Err(error);
    }
    copy_mapped_gather(&first_transfer, first_out)?;
    copy_mapped_gather(&second_transfer, second_out)
}

struct ScatterTransfer {
    _meta: UniformBufferGuard,
    bind_group: wgpu::BindGroup,
    compact_len: usize,
}

fn prepare_scatter(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    buffer: &WgpuBuffer<f32>,
    temp_compact_buf: &WgpuBuffer<f32>,
    compact_host: &[f32],
    region: MatrixRegion,
) -> Result<ScatterTransfer> {
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
    let raw_meta = device.get_uniform_buffer(WgpuDevice::byte_size::<RegionCopyMeta>(1)?)?;
    let meta = uniform_guard(device.clone(), raw_meta);
    device
        .queue()
        .write_buffer(&meta, 0, eunomia::layout::bytes_of(&region_meta(region)?));
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
                    resource: meta.as_entire_binding(),
                },
            ],
        });
    Ok(ScatterTransfer {
        _meta: meta,
        bind_group,
        compact_len,
    })
}

fn encode_scatter(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    transfer: &ScatterTransfer,
) -> Result<()> {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("hephaestus-region-scatter-pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &transfer.bind_group, &[]);
    pass.dispatch_workgroups(checked_wg_x(transfer.compact_len.div_ceil(256))?, 1, 1);
    Ok(())
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

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<RegionCopyKernel>(), TypeId::of::<f32>(), 1),
        "hephaestus-region-scatter",
        region_scatter_shader_source,
    );
    let transfer = prepare_scatter(
        device,
        &pipeline,
        buffer,
        temp_compact_buf,
        compact_host,
        region,
    )?;
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matrix-region-upload-compact"),
        });
    encode_scatter(&mut encoder, &pipeline, &transfer)?;
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

/// Scatter two compact host matrices into two regions through one encoder and
/// one queue submission.
///
/// Non-empty transfers require distinct temporary device buffers because both
/// host slices are staged before either scatter pass is encoded.
pub(crate) fn write_matrix_region_pair_compact_reusable(
    device: &WgpuDevice,
    buffer: &WgpuBuffer<f32>,
    first: MatrixRegionUpload<'_>,
    second: MatrixRegionUpload<'_>,
) -> Result<()> {
    if first.region.rows == 0 || first.region.cols == 0 {
        return write_matrix_region_compact_reusable(
            device,
            buffer,
            second.temp,
            second.host,
            second.region,
        );
    }
    if second.region.rows == 0 || second.region.cols == 0 {
        return write_matrix_region_compact_reusable(
            device,
            buffer,
            first.temp,
            first.host,
            first.region,
        );
    }
    if first.temp.aliases(second.temp) {
        return Err(HephaestusError::TransferFailed {
            message: "paired region scatter requires distinct temporary buffers".to_string(),
        });
    }

    let pipeline = cached_pipeline(
        device,
        (TypeId::of::<RegionCopyKernel>(), TypeId::of::<f32>(), 1),
        "hephaestus-region-scatter",
        region_scatter_shader_source,
    );
    let first = prepare_scatter(
        device,
        &pipeline,
        buffer,
        first.temp,
        first.host,
        first.region,
    )?;
    let second = prepare_scatter(
        device,
        &pipeline,
        buffer,
        second.temp,
        second.host,
        second.region,
    )?;
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matrix-region-pair-upload-compact"),
        });
    encode_scatter(&mut encoder, &pipeline, &first)?;
    encode_scatter(&mut encoder, &pipeline, &second)?;
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use hephaestus_core::ComputeDevice;

    use super::*;

    #[test]
    fn paired_scatter_rejects_aliased_temporary_buffers() {
        let Ok(device) = WgpuDevice::try_default("paired-scatter-alias-contract") else {
            return;
        };
        let target = device.alloc_zeroed::<f32>(4).expect("target allocation");
        let first_temp = device.alloc_zeroed::<f32>(2).expect("temporary allocation");
        let second_temp = first_temp.clone();
        let error = write_matrix_region_pair_compact_reusable(
            &device,
            &target,
            MatrixRegionUpload {
                temp: &first_temp,
                host: &[1.0, 2.0],
                region: MatrixRegion {
                    stride: 2,
                    row_start: 0,
                    col_start: 0,
                    rows: 1,
                    cols: 2,
                },
            },
            MatrixRegionUpload {
                temp: &second_temp,
                host: &[3.0, 4.0],
                region: MatrixRegion {
                    stride: 2,
                    row_start: 1,
                    col_start: 0,
                    rows: 1,
                    cols: 2,
                },
            },
        )
        .expect_err("aliased paired-scatter temporary buffers must be rejected");
        assert!(matches!(
            error,
            HephaestusError::TransferFailed { message }
                if message.contains("distinct temporary buffers")
        ));
    }
}
