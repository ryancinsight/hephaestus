//! WGPU implementation of [`hephaestus_core::SortOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK):
//! one thread per axis lane decodes its lane's base offsets with the
//! padded rank-8 odometer, then insertion-sorts the lane streaming from
//! the input, maintaining `(value, original position)` pairs directly
//! in the output slots the way the top-k seam does — no scratch array,
//! no padding. The inner bubble step swaps only on a strict comparison
//! (`>` ascending, `<` descending), which makes the sort stable: equal
//! cells never cross. WGSL does not advertise IEEE special values, so a
//! `NaN` in a lane orders it unspecifically on this backend — the
//! shared special-values clause skips `NaN`-mixed lanes here wholesale
//! by construction (ADR 0043) — while every finite edge (`-0.0` stable
//! with `+0.0`, `-inf` below everything finite) holds exactly as on
//! the host.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, SortOps, StridedView, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `SortMeta` uniform: rank-8 padded input shape, input, values,
/// and indices strides, `[in_offset, val_offset, idx_offset, _]`, and
/// `[lanes, extent, dim_lane, descending]`.
///
/// `dim_lane` is the padded lane holding `dim` (`8 - N + dim`); shapes
/// pack leading ones, so the operation rank occupies the trailing
/// lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    val_strides: [i32; 8],
    idx_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<SortMeta>() == 160);

const WGSL_SORT_META: &str = r"struct SortMeta {
    in_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    val_strides: array<vec4<i32>, 2>,
    idx_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn sort_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> smeta: SortMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out_vals: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out_idx: array<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let lane = gid.x;
    if (lane >= smeta.params.x) {{
        return;
    }}
    let extent = smeta.params.y;
    let dim_lane = smeta.params.z;
    let descending = smeta.params.w;
    var rem = lane;
    var in_base = i32(smeta.offsets.x);
    var val_base = i32(smeta.offsets.y);
    var idx_base = i32(smeta.offsets.z);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        if (p == i32(dim_lane)) {{
            continue;
        }}
        let group = p / 4;
        let lane_c = p % 4;
        let d = smeta.in_shape[group][lane_c];
        let c = rem % d;
        rem = rem / d;
        in_base = in_base + i32(c) * smeta.in_strides[group][lane_c];
        val_base = val_base + i32(c) * smeta.val_strides[group][lane_c];
        idx_base = idx_base + i32(c) * smeta.idx_strides[group][lane_c];
    }}
    let group_d = dim_lane / 4u;
    let lane_d = dim_lane % 4u;
    let in_stride = smeta.in_strides[group_d][lane_d];
    let val_stride = smeta.val_strides[group_d][lane_d];
    let idx_stride = smeta.idx_strides[group_d][lane_d];

    for (var a: u32 = 0u; a < extent; a = a + 1u) {{
        let val = input[u32(in_base + i32(a) * in_stride)];
        var pos = a;
        out_vals[u32(val_base + i32(pos) * val_stride)] = val;
        out_idx[u32(idx_base + i32(pos) * idx_stride)] = a;
        loop {{
            if (pos == 0u) {{ break; }}
            let vp = u32(val_base + i32(pos) * val_stride);
            let vq = u32(val_base + i32(pos - 1u) * val_stride);
            if (descending == 0u) {{
                if (out_vals[vq] <= out_vals[vp]) {{ break; }}
            }} else {{
                if (out_vals[vq] >= out_vals[vp]) {{ break; }}
            }}
            let ip = u32(idx_base + i32(pos) * idx_stride);
            let iq = u32(idx_base + i32(pos - 1u) * idx_stride);
            let tv = out_vals[vq];
            let ti = out_idx[iq];
            out_vals[vq] = out_vals[vp];
            out_idx[iq] = out_idx[ip];
            out_vals[vp] = tv;
            out_idx[ip] = ti;
            pos = pos - 1u;
        }}
    }}
}}
"#,
        meta = WGSL_SORT_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct SortKernel;

/// WGPU device marker implementing [`hephaestus_core::SortOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuSortOps;

impl<T> SortOps<WgpuDevice, T> for WgpuSortOps
where
    T: DialectScalar<Wgsl> + Pod + PartialOrd,
{
    type Dialect = Wgsl;

    fn sort_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        dim: usize,
        descending: bool,
        values: StridedView<'_, WgpuBuffer<T>, N>,
        indices: StridedView<'_, WgpuBuffer<u32>, N>,
    ) -> Result<()> {
        sort_into::<T, N>(
            device,
            input,
            dim,
            descending,
            values,
            indices,
            BlockWidth::DEFAULT,
        )
    }
}

/// Stably sort each lane along `dim` at a caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `dim` is out of range, the output
/// shapes disagree with the input, the values buffer aliases the input
/// buffer, a layout is unsupported, or the backend dispatch fails.
pub fn sort_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    dim: usize,
    descending: bool,
    values: StridedView<'_, WgpuBuffer<T>, N>,
    indices: StridedView<'_, WgpuBuffer<u32>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + PartialOrd,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "sort dispatch supports rank <= 8");
    }

    hephaestus_core::validate_sort_shape(
        input.layout.shape(),
        dim,
        values.layout.shape(),
        indices.layout.shape(),
    )?;

    let in_layout = input.layout;
    let val_layout = values.layout;
    let idx_layout = indices.layout;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    val_layout
        .validate_storage_len(values.buffer.len)
        .map_err(map_layout_err)?;
    idx_layout
        .validate_storage_len(indices.buffer.len)
        .map_err(map_layout_err)?;
    if !val_layout.is_injective().map_err(map_layout_err)?
        || !idx_layout.is_injective().map_err(map_layout_err)?
    {
        return Err(HephaestusError::DispatchFailed {
            message: "sort output layouts must be non-overlapping".to_string(),
        });
    }
    if values.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "sort values buffer must not alias the input buffer".to_string(),
        });
    }
    let len = val_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(());
    }

    let extent = in_layout.shape()[dim];
    let lanes = len / extent;
    let meta = SortMeta {
        in_shape: pad_shape(in_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        val_strides: pad_strides(val_layout.strides())?,
        idx_strides: pad_strides(idx_layout.strides())?,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(val_layout.offset(), "values offset")?,
            to_u32(idx_layout.offset(), "indices offset")?,
            0,
        ],
        params: [
            to_u32(lanes, "dispatch size")?,
            to_u32(extent, "sort extent")?,
            to_u32(8 - N + dim, "sort dim lane")?,
            u32::from(descending),
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<SortMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<SortKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-sort",
        || sort_shader::<T>(block_width),
    );

    let groups = workgroups(lanes, block_width)?;

    let mut entries = BindGroupEntries::with_capacity(4);
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
        resource: values.buffer.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: indices.buffer.buffer.as_entire_binding(),
    });
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-sort"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-sort"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-sort",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
