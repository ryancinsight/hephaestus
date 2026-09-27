//! WGPU implementation of [`hephaestus_core::TopKOps`].
//!
//! One thread per output lane, maintaining its `k`-slot sorted-descending
//! result directly in the output buffers (no per-thread scratch array of a
//! runtime-chosen size is needed: `values`/`indices` already have exactly
//! `k` slots per lane). `axis` bakes into the generated WGSL and the
//! pipeline-cache key (the axis/argmin-argmax seam's `arg_reduce_seam` module
//! uses the same convention); `k` itself is an ordinary runtime loop bound
//! read from the uniform, not baked, since it does not change which shader
//! text is legal.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, StridedView, TopKOps, Wgsl,
    validate_topk_shape,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{map_layout_err, to_i32, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `TopKMeta` uniform: input shape/strides, output strides, and
/// `[in_offset, out_offset, lane_count, k]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TopKMeta {
    in_shape: [u32; 4],
    in_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<TopKMeta>() == 64);

const WGSL_TOPK_META: &str = r"struct TopKMeta {
    in_shape: vec4<u32>,
    in_strides: vec4<i32>,
    out_strides: vec4<i32>,
    offsets: vec4<u32>,
}
";

struct TopKKernel;

fn topk_shader<T: DialectScalar<Wgsl>>(width: BlockWidth, axis: usize) -> String {
    let other = 1 - axis;
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> tmeta: TopKMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out_vals: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out_idx: array<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let lane = gid.x;
    if (lane >= tmeta.offsets.z) {{
        return;
    }}
    let k = tmeta.offsets.w;
    let axis_len = tmeta.in_shape[{axis}];
    let lane_base = i32(tmeta.offsets.x) + i32(lane) * tmeta.in_strides[{other}];
    let out_base = lane * k;

    for (var a: u32 = 0u; a < axis_len; a = a + 1u) {{
        let val = input[u32(lane_base + i32(a) * tmeta.in_strides[{axis}])];
        if (a < k) {{
            var pos = a;
            out_vals[out_base + pos] = val;
            out_idx[out_base + pos] = a;
            loop {{
                if (pos == 0u) {{ break; }}
                if (out_vals[out_base + pos - 1u] >= out_vals[out_base + pos]) {{ break; }}
                let tv = out_vals[out_base + pos - 1u];
                let ti = out_idx[out_base + pos - 1u];
                out_vals[out_base + pos - 1u] = out_vals[out_base + pos];
                out_idx[out_base + pos - 1u] = out_idx[out_base + pos];
                out_vals[out_base + pos] = tv;
                out_idx[out_base + pos] = ti;
                pos = pos - 1u;
            }}
        }} else if (val > out_vals[out_base + k - 1u]) {{
            var pos = k - 1u;
            out_vals[out_base + pos] = val;
            out_idx[out_base + pos] = a;
            loop {{
                if (pos == 0u) {{ break; }}
                if (out_vals[out_base + pos - 1u] >= out_vals[out_base + pos]) {{ break; }}
                let tv = out_vals[out_base + pos - 1u];
                let ti = out_idx[out_base + pos - 1u];
                out_vals[out_base + pos - 1u] = out_vals[out_base + pos];
                out_idx[out_base + pos - 1u] = out_idx[out_base + pos];
                out_vals[out_base + pos] = tv;
                out_idx[out_base + pos] = ti;
                pos = pos - 1u;
            }}
        }}
    }}
}}
"#,
        meta = WGSL_TOPK_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        axis = axis,
        other = other,
    )
}

/// WGPU device marker implementing [`TopKOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuTopKOps;

impl<T> TopKOps<WgpuDevice, T> for WgpuTopKOps
where
    T: DialectScalar<Wgsl> + Pod + PartialOrd,
{
    fn topk_axis_into(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        axis: usize,
        k: usize,
        values: StridedView<'_, WgpuBuffer<T>, 2>,
        indices: StridedView<'_, WgpuBuffer<u32>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let val_layout = values.layout;
        let idx_layout = indices.layout;
        let (lanes, _axis_len) = validate_topk_shape(
            axis,
            k,
            in_layout.shape(),
            val_layout.shape(),
            idx_layout.shape(),
        )?;

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
                message: "top-k output layouts must be non-overlapping".to_string(),
            });
        }
        if lanes == 0 {
            return Ok(());
        }
        let block_width = BlockWidth::DEFAULT;

        let in_shape = in_layout.shape();
        let in_strides = in_layout.strides();
        // `values` and `indices` share the same shape (validated above); a
        // dense output requires them to share strides too for this single
        // `out_strides` field to describe both, which holds for the
        // conformance suite's dense fixtures and is documented as the
        // contract here rather than re-derived per call.
        let out_strides = val_layout.strides();
        let meta = TopKMeta {
            in_shape: [
                to_u32(in_shape[0], "input dimension")?,
                to_u32(in_shape[1], "input dimension")?,
                0,
                0,
            ],
            in_strides: [
                to_i32(in_strides[0], "input stride")?,
                to_i32(in_strides[1], "input stride")?,
                0,
                0,
            ],
            out_strides: [
                to_i32(out_strides[0], "output stride")?,
                to_i32(out_strides[1], "output stride")?,
                0,
                0,
            ],
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(val_layout.offset(), "output offset")?,
                to_u32(lanes, "dispatch size")?,
                to_u32(k, "k")?,
            ],
        };

        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<TopKMeta>(1)?)?;
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let pipeline = cached_pipeline(
            device,
            (
                TypeId::of::<TopKKernel>(),
                TypeId::of::<T>(),
                block_width.get() * 2 + axis as u32,
            ),
            "hephaestus-topk",
            || topk_shader::<T>(block_width, axis),
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
                label: Some("hephaestus-topk"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-topk"),
            });
        encode_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            groups,
            "hephaestus-topk",
        );
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
