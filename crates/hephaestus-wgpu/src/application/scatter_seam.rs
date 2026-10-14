//! WGPU implementation of [`hephaestus_core::ScatterOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK): one
//! thread per output cell first copies its input cell, then scans every
//! source cell in ascending row-major order and adds the ones whose index
//! maps to it. The scan is output-parallel because repeated indices would
//! race an input-parallel scatter (and WGSL has no floating-point
//! atomics); the ascending scan keeps every backend bitwise-identical on
//! floating-point sums, exactly as the embedding scatter-add does. `+`
//! applies to every [`DialectScalar<Wgsl>`](hephaestus_core::DialectScalar)
//! type — numeric scalars and numeric vectors alike — so the kernel needs
//! no further scalar bound.
//!
//! The copy and the scan are two launches, not one: the scan's four
//! storage buffers (indices, source, output, error flag) already fill the
//! portable `max_storage_buffers_per_shader_stage` limit of four, so the
//! input cannot join the same bind group. The copy kernel runs first and
//! the scan seeds its accumulator from the copied output cell — each
//! thread still touches only its own cell, so no cross-thread sharing
//! appears.
//!
//! A source cell maps to its thread's cell exactly when the output offset
//! recomputed from its coordinates equals the thread's own offset, which
//! is a sound comparison only because validation proves the output layout
//! injective. The index *values* are device-resident data, so an
//! out-of-range value cannot be caught before dispatch the way a
//! host-known shape mismatch can. The kernel detects it and reports it
//! through a one-`u32` atomic error flag, downloaded after the dispatch
//! completes, rather than clamping or wrapping (error-handling restraint:
//! a data-dependent fault is surfaced, never silently masked). A faulting
//! update maps to no cell and is skipped; the output is left partially
//! accumulated, which is irrelevant once the call returns an error.
//!
//! When the index holds no updates, a smaller copy kernel runs instead of
//! the scan: binding the empty index/source buffers would violate the
//! nonzero minimum binding size, and there is nothing to accumulate.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, ScatterOps, StridedView,
    Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `ScatterNdMeta` uniform: rank-8 padded output and index shapes,
/// index, source, and output strides, `[index_offset, src_offset,
/// output_offset, _]`, and `[output_len, src_len, dim_lane, dim_extent]`.
///
/// `dim_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes. The
/// input needs no entry: the copy kernel runs first and the scan seeds
/// from the copied output cell.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterNdMeta {
    out_shape: [u32; 8],
    idx_shape: [u32; 8],
    idx_strides: [i32; 8],
    src_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<ScatterNdMeta>() == 192);

const WGSL_SCATTER_ND_META: &str = r"struct ScatterNdMeta {
    out_shape: array<vec4<u32>, 2>,
    idx_shape: array<vec4<u32>, 2>,
    idx_strides: array<vec4<i32>, 2>,
    src_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn scatter_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: ScatterNdMeta;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= smeta.params.x) {{
        return;
    }}
    let dim_lane = i32(smeta.params.z);
    let dim_extent = smeta.params.w;
    var rem = i;
    var out_off = i32(smeta.offsets.z);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = smeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        out_off = out_off + i32(c) * smeta.out_strides[group][lane];
    }}
    // Seeded from the copied output cell: the copy kernel ran first, and
    // this thread owns its cell, so the read shares with no other thread.
    var acc: {ty} = out[u32(out_off)];
    let src_len = smeta.params.y;
    for (var j: u32 = 0u; j < src_len; j = j + 1u) {{
        var srem = j;
        var idx_off = i32(smeta.offsets.x);
        var src_off = i32(smeta.offsets.y);
        var mapped = i32(smeta.offsets.z);
        for (var p: i32 = 7; p >= 0; p = p - 1) {{
            let group = p / 4;
            let lane = p % 4;
            let idx_dim = smeta.idx_shape[group][lane];
            let c = srem % idx_dim;
            srem = srem / idx_dim;
            idx_off = idx_off + i32(c) * smeta.idx_strides[group][lane];
            src_off = src_off + i32(c) * smeta.src_strides[group][lane];
            if (p != dim_lane) {{
                mapped = mapped + i32(c) * smeta.out_strides[group][lane];
            }}
        }}
        let g = indices[u32(idx_off)];
        if (g >= dim_extent) {{
            atomicStore(&error_flag, 1u);
            continue;
        }}
        mapped = mapped + i32(g) * smeta.out_strides[dim_lane / 4][dim_lane % 4];
        if (mapped == out_off) {{
            acc = acc + src[u32(src_off)];
        }}
    }}
    out[u32(out_off)] = acc;
}}
"#,
        meta = WGSL_SCATTER_ND_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

/// WGSL `ScatterCopyMeta` uniform: rank-8 padded output shape, input and
/// output strides, and `[input_offset, output_offset, len]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterCopyMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<ScatterCopyMeta>() == 112);

const WGSL_SCATTER_COPY_META: &str = r"struct ScatterCopyMeta {
    out_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
}
";

fn scatter_copy_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> cmeta: ScatterCopyMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= cmeta.offsets.z) {{
        return;
    }}
    var rem = i;
    var in_off = i32(cmeta.offsets.x);
    var out_off = i32(cmeta.offsets.y);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = cmeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + i32(c) * cmeta.in_strides[group][lane];
        out_off = out_off + i32(c) * cmeta.out_strides[group][lane];
    }}
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_SCATTER_COPY_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<ScatterNdMeta>(), 192);
        assert_eq!(core::mem::size_of::<ScatterCopyMeta>(), 112);
    }
}
