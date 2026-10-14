//! WGPU implementation of [`hephaestus_core::IndexSelectOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK): one
//! thread per output cell decodes its output coordinate, reads the `u32`
//! index at its own `dim` coordinate, substitutes it into the `dim` lane,
//! and copies the addressed input cell. No bounds check on the coordinate
//! is needed because validation proves the output shape against the input
//! shape and the index length exactly — but the index *values* are
//! device-resident data, so an out-of-range value cannot be caught before
//! dispatch the way a host-known shape mismatch can. The kernel detects
//! it and reports it through a one-`u32` atomic error flag, downloaded
//! after the dispatch completes, rather than clamping or wrapping
//! (error-handling restraint: a data-dependent fault is surfaced, never
//! silently masked), exactly as the embedding seam does. The copy is
//! skipped for any lane that trips the flag; its output cell is left at
//! whatever `output` held on entry, which is irrelevant once the call
//! returns an error.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, IndexSelectOps, Result, StridedView,
    Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `IndexSelectMeta` uniform: rank-8 padded output shape, input,
/// index, and output strides, `[input_offset, index_offset, output_offset,
/// len]`, and `[dim_lane, dim_extent, _, _]`.
///
/// `dim_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes pack
/// leading ones, so the operation rank occupies the trailing lanes. The
/// index rank is concrete (`1`), so its padded strides hold the real
/// stride in the trailing lane and zeros elsewhere.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexSelectMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    idx_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<IndexSelectMeta>() == 160);

const WGSL_INDEX_SELECT_META: &str = r"struct IndexSelectMeta {
    out_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    idx_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn index_select_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: IndexSelectMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= smeta.offsets.w) {{
        return;
    }}
    let dim_lane = i32(smeta.params.x);
    var rem = i;
    var in_off = i32(smeta.offsets.x);
    var out_off = i32(smeta.offsets.z);
    var slot = 0u;
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = smeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + i32(c) * smeta.in_strides[group][lane];
        out_off = out_off + i32(c) * smeta.out_strides[group][lane];
        if (p == dim_lane) {{
            slot = c;
        }}
    }}
    // The rank-1 index stride lives in the trailing padded lane; the
    // leading lanes pad zeros.
    let idx_off = i32(smeta.offsets.y) + i32(slot) * smeta.idx_strides[1][3];
    let g = indices[u32(idx_off)];
    if (g >= smeta.params.y) {{
        atomicStore(&error_flag, 1u);
        return;
    }}
    let stride_group = dim_lane / 4;
    let stride_lane = dim_lane % 4;
    let dim_stride = smeta.in_strides[stride_group][stride_lane];
    in_off = in_off - i32(slot) * dim_stride + i32(g) * dim_stride;
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_INDEX_SELECT_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct IndexSelectKernel;

/// WGPU device marker implementing [`hephaestus_core::IndexSelectOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuIndexSelectOps;

impl<T> IndexSelectOps<WgpuDevice, T> for WgpuIndexSelectOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn index_select_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, WgpuBuffer<u32>, 1>,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        index_select_into::<T, N>(device, input, dim, index, output, BlockWidth::DEFAULT)
    }
}

/// Write `output[i] = input[i with dim := index[i_dim]]` at a
/// caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `dim` is out of range, a non-`dim`
/// dimension disagrees, the output's `dim` extent is not the index length,
/// any index value is out of range for the input's `dim` extent, the
/// output buffer aliases the input buffer, a layout is unsupported, or the
/// backend dispatch fails.
pub fn index_select_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    dim: usize,
    index: StridedView<'_, WgpuBuffer<u32>, 1>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "index_select dispatch supports rank <= 8"
        );
    }

    hephaestus_core::validate_index_select_shape(
        input.layout.shape(),
        dim,
        index.layout.shape()[0],
        output.layout.shape(),
    )?;

    let in_layout = input.layout;
    let idx_layout = index.layout;
    let out_layout = output.layout;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    idx_layout
        .validate_storage_len(index.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "index_select output layout must be non-overlapping".to_string(),
        });
    }
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "index_select output buffer must not alias the input buffer".to_string(),
        });
    }
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        // Zero cells read no index, so no out-of-range value can fault —
        // even when the index itself is nonempty against an empty
        // non-`dim` lane.
        return Ok(());
    }

    let dim_extent = in_layout.shape()[dim];
    if dim_extent == 0 {
        // No input cell exists for any thread to read, so the kernel could
        // not even bind its source — but every index is trivially out of
        // range for an empty selection dimension.
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "index_select encountered an index out of range for dim {dim} extent {dim_extent}",
            ),
        });
    }

    let meta = IndexSelectMeta {
        out_shape: pad_shape(out_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        idx_strides: pad_strides(idx_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(idx_layout.offset(), "index offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
        params: [
            to_u32(8 - N + dim, "index_select dim lane")?,
            to_u32(dim_extent, "index_select dim extent")?,
            0,
            0,
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<IndexSelectMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let error_flag = device.alloc_zeroed::<u32>(1)?;

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<IndexSelectKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-index-select",
        || index_select_shader::<T>(block_width),
    );

    let groups = workgroups(len, block_width)?;

    let bind_group = {
        let mut entries = BindGroupEntries::with_capacity(5);
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
            resource: index.buffer.buffer.as_entire_binding(),
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
                label: Some("hephaestus-index-select"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
    };

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-index-select"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-index-select",
    );
    device.queue().submit(Some(encoder.finish()));

    let mut flag_host = [0u32; 1];
    device.download(&error_flag, &mut flag_host)?;
    if flag_host[0] != 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "index_select encountered an index out of range for dim {dim} extent {dim_extent}",
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_select_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<IndexSelectMeta>(), 160);
    }
}
