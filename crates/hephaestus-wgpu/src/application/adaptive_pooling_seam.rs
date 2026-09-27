//! WGPU implementation of [`hephaestus_core::AdaptivePoolingOps`].
//!
//! One thread per `(lane, out_idx)` pair (flat-dispatched over
//! `lanes * out_len`, matching `interpolation_seam`'s dispatch shape)
//! computes its own `[start, end)` window and reduces it directly — no
//! shared state between threads. `mode` and `axis` bake into the generated
//! WGSL at pipeline-cache-key granularity.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    AdaptivePoolingMode, AdaptivePoolingOps, BlockWidth, DialectScalar, HephaestusError, Result,
    StridedView, Wgsl, validate_adaptive_pooling_shape,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{map_layout_err, to_i32, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `AdaptivePoolingMeta` uniform. Every field is a full `vec4` for the
/// same reason `ArgReduceMeta` documents (ADR 0063).
///
/// `in_strides`/`out_strides` use lanes `[axis, other]`; lanes `[2]`/`[3]`
/// are unused padding. `offsets` is `[in_offset, out_offset, 0, 0]`.
/// `params` is `[in_len, out_len, lanes, 0]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdaptivePoolingMeta {
    in_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<AdaptivePoolingMeta>() == 64);

const WGSL_ADAPTIVE_POOLING_META: &str = r"struct AdaptivePoolingMeta {
    in_strides: vec4<i32>,
    out_strides: vec4<i32>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

struct AdaptivePoolingKernel;

fn adaptive_pooling_shader<T: DialectScalar<Wgsl>>(
    width: BlockWidth,
    mode: AdaptivePoolingMode,
) -> String {
    let ty = T::TYPE_TOKEN;
    let reduce_body = match mode {
        AdaptivePoolingMode::Average => format!(
            r"    var acc: {ty} = {ty}(0.0);
    for (var i: u32 = start; i < end; i = i + 1u) {{
        acc = acc + input[u32(in_base + i32(i) * ameta.in_strides.x)];
    }}
    let value = acc / {ty}(end - start);"
        ),
        AdaptivePoolingMode::Maximum => format!(
            r"    var acc: {ty} = input[u32(in_base + i32(start) * ameta.in_strides.x)];
    for (var i: u32 = start + 1u; i < end; i = i + 1u) {{
        let candidate = input[u32(in_base + i32(i) * ameta.in_strides.x)];
        if (candidate > acc) {{
            acc = candidate;
        }}
    }}
    let value = acc;"
        ),
    };
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> ameta: AdaptivePoolingMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let in_len = ameta.params.x;
    let out_len = ameta.params.y;
    let lanes = ameta.params.z;
    let total = lanes * out_len;
    if (gid.x >= total) {{
        return;
    }}
    let lane = gid.x / out_len;
    let out_idx = gid.x % out_len;

    let start = out_idx * in_len / out_len;
    let end = ((out_idx + 1u) * in_len + out_len - 1u) / out_len;
    let in_base = i32(ameta.offsets.x) + i32(lane) * ameta.in_strides.y;

{reduce_body}

    let out_off = i32(ameta.offsets.y) + i32(lane) * ameta.out_strides.y
        + i32(out_idx) * ameta.out_strides.x;
    out[u32(out_off)] = value;
}}
"#,
        meta = WGSL_ADAPTIVE_POOLING_META,
        ty = ty,
        wg = width.get(),
        reduce_body = reduce_body,
    )
}

/// WGPU device marker implementing [`AdaptivePoolingOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuAdaptivePoolingOps;

impl<T> AdaptivePoolingOps<WgpuDevice, T> for WgpuAdaptivePoolingOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn adaptive_pool_axis_into(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        axis: usize,
        mode: AdaptivePoolingMode,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let out_layout = output.layout;
        let (lanes, in_len, out_len) =
            validate_adaptive_pooling_shape(axis, in_layout.shape(), out_layout.shape())?;

        in_layout
            .validate_storage_len(input.buffer.len)
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len)
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "adaptive-pooling output layout must be non-overlapping".to_string(),
            });
        }
        if lanes == 0 {
            return Ok(());
        }
        let block_width = BlockWidth::DEFAULT;
        let other = 1 - axis;

        let in_strides = in_layout.strides();
        let out_strides = out_layout.strides();
        let meta = AdaptivePoolingMeta {
            in_strides: [
                to_i32(in_strides[axis], "input stride")?,
                to_i32(in_strides[other], "input stride")?,
                0,
                0,
            ],
            out_strides: [
                to_i32(out_strides[axis], "output stride")?,
                to_i32(out_strides[other], "output stride")?,
                0,
                0,
            ],
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                to_u32(out_layout.offset(), "output offset")?,
                0,
                0,
            ],
            params: [
                to_u32(in_len, "input axis length")?,
                to_u32(out_len, "output axis length")?,
                to_u32(lanes, "dispatch size")?,
                0,
            ],
        };

        let meta_buffer =
            device.get_uniform_buffer(WgpuDevice::byte_size::<AdaptivePoolingMeta>(1)?)?;
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let pipeline = cached_pipeline(
            device,
            (
                TypeId::of::<AdaptivePoolingKernel>(),
                TypeId::of::<T>(),
                block_width.get() * 2 + matches!(mode, AdaptivePoolingMode::Maximum) as u32,
            ),
            "hephaestus-adaptive-pooling",
            || adaptive_pooling_shader::<T>(block_width, mode),
        );

        let total =
            lanes
                .checked_mul(out_len)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: "adaptive-pooling dispatch size overflows".to_string(),
                })?;
        let groups = workgroups(total, block_width)?;
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
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-adaptive-pooling"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-adaptive-pooling"),
            });
        encode_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            groups,
            "hephaestus-adaptive-pooling",
        );
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
