//! Host validation and WGPU compute dispatch for `scatter_add`.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, ScatterOps, StridedView,
    Wgsl,
};

use super::shader::{ScatterCopyMeta, ScatterNdMeta, scatter_copy_shader, scatter_shader};
use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

struct ScatterNdKernel;
struct ScatterCopyKernel;

/// WGPU device marker implementing [`hephaestus_core::ScatterOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuScatterOps;

impl<T> ScatterOps<WgpuDevice, T> for WgpuScatterOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn scatter_add_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, WgpuBuffer<u32>, N>,
        src: StridedView<'_, WgpuBuffer<T>, N>,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        scatter_add_into::<T, N>(device, input, dim, index, src, output, BlockWidth::DEFAULT)
    }
}

/// Copy `input` into `output`, then accumulate `src[i]` into
/// `output[i with dim := index[i]]` in ascending row-major source order,
/// at a caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `dim` is out of range, index and
/// source shapes disagree, a non-`dim` index extent exceeds the input's,
/// the output shape does not match the input shape, any index value is
/// out of range for the input's `dim` extent, the output buffer aliases
/// the input or source buffer, a layout is unsupported, or the backend
/// dispatch fails.
pub fn scatter_add_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    dim: usize,
    index: StridedView<'_, WgpuBuffer<u32>, N>,
    src: StridedView<'_, WgpuBuffer<T>, N>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "scatter dispatch supports rank <= 8");
    }

    hephaestus_core::validate_scatter_shape(
        input.layout.shape(),
        dim,
        index.layout.shape(),
        src.layout.shape(),
        output.layout.shape(),
    )?;

    let in_layout = input.layout;
    let idx_layout = index.layout;
    let src_layout = src.layout;
    let out_layout = output.layout;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    idx_layout
        .validate_storage_len(index.buffer.len)
        .map_err(map_layout_err)?;
    src_layout
        .validate_storage_len(src.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "scatter output layout must be non-overlapping".to_string(),
        });
    }
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "scatter output buffer must not alias the input buffer".to_string(),
        });
    }
    if output.buffer.aliases(src.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "scatter output buffer must not alias the source buffer".to_string(),
        });
    }
    let out_len = out_layout.checked_size().map_err(map_layout_err)?;
    let src_len = idx_layout.checked_size().map_err(map_layout_err)?;
    if out_len == 0 {
        // An empty output holds no cells to fault on only when the index
        // is empty too: a nonempty index against an empty gather
        // dimension is trivially out of range for every update. (The
        // output shape equals the input shape, so an empty output with a
        // nonempty index proves the `dim` extent itself is empty: every
        // other input extent covers its index extent by validation.)
        if src_len == 0 {
            return Ok(());
        }
        let dim_extent = in_layout.shape()[dim];
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "scatter encountered an index out of range for dim {dim} extent {dim_extent}",
            ),
        });
    }
    if src_len == 0 {
        return scatter_copy::<T, N>(device, input, output, block_width);
    }
    scatter_copy::<T, N>(device, input, output, block_width)?;

    let dim_extent = in_layout.shape()[dim];

    let meta = ScatterNdMeta {
        out_shape: pad_shape(out_layout.shape())?,
        idx_shape: pad_shape(idx_layout.shape())?,
        idx_strides: pad_strides(idx_layout.strides())?,
        src_strides: pad_strides(src_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(idx_layout.offset(), "index offset")?,
            to_u32(src_layout.offset(), "src offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            0,
        ],
        params: [
            to_u32(out_len, "dispatch size")?,
            to_u32(src_len, "scan size")?,
            to_u32(8 - N + dim, "scatter dim lane")?,
            to_u32(dim_extent, "scatter dim extent")?,
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<ScatterNdMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let error_flag = device.alloc_zeroed::<u32>(1)?;

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<ScatterNdKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-scatter",
        || scatter_shader::<T>(block_width),
    );

    let groups = workgroups(out_len, block_width)?;

    let bind_group = {
        let mut entries = BindGroupEntries::with_capacity(5);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: meta_buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: index.buffer.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: src.buffer.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: output.buffer.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: error_flag.buffer.as_entire_binding(),
        });
        device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-scatter"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
    };

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-scatter"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-scatter",
    );
    device.queue().submit(Some(encoder.finish()));

    let mut flag_host = [0u32; 1];
    device.download(&error_flag, &mut flag_host)?;
    if flag_host[0] != 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "scatter encountered an index out of range for dim {dim} extent {dim_extent}",
            ),
        });
    }
    Ok(())
}

/// Copy `input` into `output` when the index holds no updates.
fn scatter_copy<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    let in_layout = input.layout;
    let out_layout = output.layout;
    let len = out_layout.checked_size().map_err(map_layout_err)?;

    let meta = ScatterCopyMeta {
        out_shape: pad_shape(out_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
            0,
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<ScatterCopyMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<ScatterCopyKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-scatter-copy",
        || scatter_copy_shader::<T>(block_width),
    );

    let groups = workgroups(len, block_width)?;

    let bind_group = {
        let mut entries = BindGroupEntries::with_capacity(3);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: meta_buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: input.buffer.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: output.buffer.buffer.as_entire_binding(),
        });
        device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-scatter-copy"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
    };

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-scatter-copy"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-scatter-copy",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
