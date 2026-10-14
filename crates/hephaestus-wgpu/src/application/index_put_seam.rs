//! WGPU implementation of [`hephaestus_core::IndexPutOps`].
//!
//! One packed-metadata layout serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK),
//! with two update kernels behind the `accumulate` flag. Both run after a
//! copy kernel first writes the base input into the output:
//!
//! * Replace mode is input-parallel: one thread per values cell reads the
//!   index at its own `dim` coordinate, substitutes it into the `dim` lane,
//!   and overwrites the addressed output cell. Duplicate indices race by
//!   design — `torch.index_put` leaves duplicate replace order undefined,
//!   so no clause pins it.
//! * Accumulate mode is output-parallel: one thread per output cell seeds
//!   its accumulator from the copied base cell, then scans every selection
//!   in ascending order and adds the values slices whose index maps to it.
//!   The scan is output-parallel because repeated indices would race an
//!   input-parallel scatter (and WGSL has no floating-point atomics); the
//!   ascending scan keeps every backend bitwise-identical on
//!   floating-point sums, exactly as the scatter seam does. `+` applies to
//!   every [`DialectScalar<Wgsl>`](hephaestus_core::DialectScalar) type —
//!   numeric scalars and numeric vectors alike — so the kernel needs no
//!   further scalar bound.
//!
//! The copy and the update are two launches, not one: the update's four
//! storage buffers (indices, values, output, error flag) already fill the
//! portable `max_storage_buffers_per_shader_stage` limit of four, so the
//! input cannot join the same bind group. Each thread still touches only
//! its own cell, so no cross-thread sharing appears.
//!
//! The index *values* are device-resident data, so an out-of-range value
//! cannot be caught before dispatch the way a host-known shape mismatch
//! can. The kernel detects it and reports it through a one-`u32` atomic
//! error flag, downloaded after the dispatch completes, rather than
//! clamping or wrapping (error-handling restraint: a data-dependent fault
//! is surfaced, never silently masked). A faulting update is skipped; the
//! output is left partially written, which is irrelevant once the call
//! returns an error.
//!
//! When the index holds no updates, a smaller copy kernel runs instead of
//! the update: binding the empty index/values buffers would violate the
//! nonzero minimum binding size, and there is nothing to write.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, IndexPutOps, Result, StridedView,
    Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `IndexPutNdMeta` uniform: rank-8 padded output and values shapes,
/// index, values, and output strides, `[index_offset, values_offset,
/// output_offset, _]`, and `[dispatch_len, sel_len, sel_lane, dim_extent]`.
///
/// `sel_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes. The
/// rank-1 index stride packs into the trailing lane (`7`). The input needs
/// no entry: the copy kernel runs first and the update seeds from the
/// copied output cell.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexPutNdMeta {
    out_shape: [u32; 8],
    vals_shape: [u32; 8],
    idx_strides: [i32; 8],
    vals_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<IndexPutNdMeta>() == 192);

const WGSL_INDEX_PUT_ND_META: &str = r"struct IndexPutNdMeta {
    out_shape: array<vec4<u32>, 2>,
    vals_shape: array<vec4<u32>, 2>,
    idx_strides: array<vec4<i32>, 2>,
    vals_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn index_put_replace_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: IndexPutNdMeta;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= smeta.params.x) {{
        return;
    }}
    let sel_lane = i32(smeta.params.z);
    var rem = i;
    var vals_off = i32(smeta.offsets.y);
    var out_off = i32(smeta.offsets.z);
    var slot = 0u;
    var dim_coord = 0u;
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let vals_dim = smeta.vals_shape[group][lane];
        let c = rem % vals_dim;
        rem = rem / vals_dim;
        vals_off = vals_off + i32(c) * smeta.vals_strides[group][lane];
        out_off = out_off + i32(c) * smeta.out_strides[group][lane];
        if (p == sel_lane) {{
            slot = c;
            dim_coord = c;
        }}
    }}
    // The rank-1 index stride lives in the trailing padded lane; the
    // leading lanes pad zeros.
    let idx_off = i32(smeta.offsets.x) + i32(slot) * smeta.idx_strides[1][3];
    let g = indices[u32(idx_off)];
    if (g >= smeta.params.w) {{
        atomicStore(&error_flag, 1u);
        return;
    }}
    let dim_stride = smeta.out_strides[sel_lane / 4][sel_lane % 4];
    out_off = out_off - i32(dim_coord) * dim_stride + i32(g) * dim_stride;
    out[u32(out_off)] = values[u32(vals_off)];
}}
"#,
        meta = WGSL_INDEX_PUT_ND_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

fn index_put_accumulate_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: IndexPutNdMeta;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= smeta.params.x) {{
        return;
    }}
    let sel_lane = i32(smeta.params.z);
    let dim_extent = smeta.params.w;
    let dim_stride = smeta.out_strides[sel_lane / 4][sel_lane % 4];
    let vals_dim_stride = smeta.vals_strides[sel_lane / 4][sel_lane % 4];
    var rem = i;
    var out_off = i32(smeta.offsets.z);
    // The values offset with the `dim` lane zeroed: each scan step adds
    // back `s * vals_dim_stride`.
    var vals_base = i32(smeta.offsets.y);
    var wanted = 0u;
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = smeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        out_off = out_off + i32(c) * smeta.out_strides[group][lane];
        if (p != sel_lane) {{
            vals_base = vals_base + i32(c) * smeta.vals_strides[group][lane];
        }} else {{
            wanted = c;
        }}
    }}
    // Seeded from the copied output cell: the copy kernel ran first, and
    // this thread owns its cell, so the read shares with no other thread.
    var acc: {ty} = out[u32(out_off)];
    let sel_len = smeta.params.y;
    let idx_base = i32(smeta.offsets.x);
    let idx_stride = smeta.idx_strides[1][3];
    for (var s: u32 = 0u; s < sel_len; s = s + 1u) {{
        let g = indices[u32(idx_base + i32(s) * idx_stride)];
        if (g >= dim_extent) {{
            atomicStore(&error_flag, 1u);
            continue;
        }}
        if (g == wanted) {{
            acc = acc + values[u32(vals_base + i32(s) * vals_dim_stride)];
        }}
    }}
    out[u32(out_off)] = acc;
}}
"#,
        meta = WGSL_INDEX_PUT_ND_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

/// WGSL `IndexPutCopyMeta` uniform: rank-8 padded output shape, input and
/// output strides, and `[input_offset, output_offset, len]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexPutCopyMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<IndexPutCopyMeta>() == 112);

const WGSL_INDEX_PUT_COPY_META: &str = r"struct IndexPutCopyMeta {
    out_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
}
";

fn index_put_copy_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> cmeta: IndexPutCopyMeta;
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
        meta = WGSL_INDEX_PUT_COPY_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct IndexPutReplaceKernel;
struct IndexPutAccumulateKernel;
struct IndexPutCopyKernel;

/// WGPU device marker implementing [`hephaestus_core::IndexPutOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuIndexPutOps;

impl<T> IndexPutOps<WgpuDevice, T> for WgpuIndexPutOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn index_put_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, WgpuBuffer<u32>, 1>,
        values: StridedView<'_, WgpuBuffer<T>, N>,
        accumulate: bool,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        index_put_into::<T, N>(
            device,
            input,
            dim,
            index,
            values,
            accumulate,
            output,
            BlockWidth::DEFAULT,
        )
    }
}

/// Copy `input` into `output`, then scatter `values` at `index` along
/// `dim` — replacing, or accumulating in ascending selection order — at a
/// caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `dim` is out of range, the output
/// shape is not the input shape, a values dimension disagrees, the values'
/// `dim` extent is not the index length, any index value is out of range
/// for the input's `dim` extent, the output buffer aliases the input or
/// values buffer, a layout is unsupported, or the backend dispatch fails.
#[allow(clippy::too_many_arguments)]
pub fn index_put_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    dim: usize,
    index: StridedView<'_, WgpuBuffer<u32>, 1>,
    values: StridedView<'_, WgpuBuffer<T>, N>,
    accumulate: bool,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "index_put dispatch supports rank <= 8"
        );
    }

    hephaestus_core::validate_index_put_shape(
        input.layout.shape(),
        dim,
        index.layout.shape()[0],
        values.layout.shape(),
        output.layout.shape(),
    )?;

    let in_layout = input.layout;
    let idx_layout = index.layout;
    let vals_layout = values.layout;
    let out_layout = output.layout;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    idx_layout
        .validate_storage_len(index.buffer.len)
        .map_err(map_layout_err)?;
    vals_layout
        .validate_storage_len(values.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "index_put output layout must be non-overlapping".to_string(),
        });
    }
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "index_put output buffer must not alias the input buffer".to_string(),
        });
    }
    if output.buffer.aliases(values.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "index_put output buffer must not alias the values buffer".to_string(),
        });
    }
    let out_len = out_layout.checked_size().map_err(map_layout_err)?;
    let sel_len = idx_layout.checked_size().map_err(map_layout_err)?;
    if out_len == 0 {
        // An empty output holds no cells to fault on only when the update
        // is empty too — but `sel_len > 0` against an empty non-`dim`
        // extent is still vacuous (the values hold no cells), matching the
        // host's early-out. Only an empty `dim` extent with selections
        // proves every index trivially out of range.
        if sel_len == 0 {
            return Ok(());
        }
        let mut vacuous = false;
        for (d, &extent) in vals_layout.shape().iter().enumerate() {
            if d != dim && extent == 0 {
                vacuous = true;
            }
        }
        if vacuous {
            return Ok(());
        }
        let dim_extent = in_layout.shape()[dim];
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "index_put encountered an index out of range for dim {dim} extent {dim_extent}",
            ),
        });
    }
    if sel_len == 0 {
        return index_put_copy::<T, N>(device, input, output, block_width);
    }
    index_put_copy::<T, N>(device, input, output, block_width)?;

    let dim_extent = in_layout.shape()[dim];
    let vals_len = vals_layout.checked_size().map_err(map_layout_err)?;

    let meta = IndexPutNdMeta {
        out_shape: pad_shape(out_layout.shape())?,
        vals_shape: pad_shape(vals_layout.shape())?,
        idx_strides: pad_strides(idx_layout.strides())?,
        vals_strides: pad_strides(vals_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(idx_layout.offset(), "index offset")?,
            to_u32(vals_layout.offset(), "values offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            0,
        ],
        params: [
            to_u32(if accumulate { out_len } else { vals_len }, "dispatch size")?,
            to_u32(sel_len, "selection count")?,
            to_u32(8 - N + dim, "index_put dim lane")?,
            to_u32(dim_extent, "index_put dim extent")?,
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<IndexPutNdMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let error_flag = device.alloc_zeroed::<u32>(1)?;

    let (marker, label, source) = if accumulate {
        (
            TypeId::of::<IndexPutAccumulateKernel>(),
            "hephaestus-index_put-accumulate",
            index_put_accumulate_shader::<T>(block_width),
        )
    } else {
        (
            TypeId::of::<IndexPutReplaceKernel>(),
            "hephaestus-index_put-replace",
            index_put_replace_shader::<T>(block_width),
        )
    };
    let pipeline = cached_pipeline(
        device,
        (marker, TypeId::of::<T>(), block_width.get()),
        label,
        || source,
    );

    let groups = workgroups(if accumulate { out_len } else { vals_len }, block_width)?;

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
            resource: values.buffer.buffer.as_entire_binding(),
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
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
    };

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    encode_compute_pass(&mut encoder, &pipeline, &bind_group, groups, label);
    device.queue().submit(Some(encoder.finish()));

    let mut flag_host = [0u32; 1];
    device.download(&error_flag, &mut flag_host)?;
    if flag_host[0] != 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "index_put encountered an index out of range for dim {dim} extent {dim_extent}",
            ),
        });
    }
    Ok(())
}

/// Copy `input` into `output` when the index holds no updates.
fn index_put_copy<T, const N: usize>(
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

    let meta = IndexPutCopyMeta {
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
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<IndexPutCopyMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<IndexPutCopyKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-index_put-copy",
        || index_put_copy_shader::<T>(block_width),
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
                label: Some("hephaestus-index_put-copy"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
    };

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-index_put-copy"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-index_put-copy",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
