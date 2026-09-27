//! WGPU implementation of [`hephaestus_core::ArgReduceOps`].
//!
//! One thread per output lane scans the full reduced axis in a plain loop —
//! parallelism is across lanes, not within one lane's scan. This is the
//! correctness-first cut (mirrors the host reference's tie-break rule
//! exactly); a lane-internal tree reduction is a follow-up performance item
//! if a workload needs it (`axis_len` large relative to lane count).
//!
//! `axis` and the comparison direction (argmax vs argmin) are baked into the
//! generated WGSL at pipeline-cache-key granularity, so each of the four
//! combinations (2 axes x {max, min}) monomorphizes to its own kernel with no
//! per-invocation branch.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    ArgReduceOps, BlockWidth, DialectScalar, HephaestusError, Result, StridedView, Wgsl,
    validate_arg_reduce_shape,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{map_layout_err, to_i32, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `ArgReduceMeta` uniform. Every field is a full `vec4` even though
/// only the first two lanes carry real shape/stride data: WGSL's uniform
/// address space requires each member aligned to its own size, and `vec4`
/// needs 16-byte alignment — a tighter `vec2` packing here would leave
/// `offsets` at a 24-byte offset, which the WGSL compiler would silently pad
/// to 32, desynchronizing this struct's byte layout from the Rust side's
/// (the bug this shape avoids: every cumulative offset is already a multiple
/// of 16, so no compiler-inserted padding can appear).
///
/// `in_shape`/`in_strides`/`out_strides` use lanes `[0]`/`[1]`; lanes
/// `[2]`/`[3]` are unused padding. `offsets` is
/// `[in_offset, out_offset, lane_count, _pad]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ArgReduceMeta {
    in_shape: [u32; 4],
    in_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<ArgReduceMeta>() == 64);

const WGSL_ARG_REDUCE_META: &str = r"struct ArgReduceMeta {
    in_shape: vec4<u32>,
    in_strides: vec4<i32>,
    out_strides: vec4<i32>,
    offsets: vec4<u32>,
}
";

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ArgReduceDirection {
    Max,
    Min,
}

struct ArgReduceKernel;

fn arg_reduce_shader<T: DialectScalar<Wgsl>>(
    width: BlockWidth,
    axis: usize,
    direction: ArgReduceDirection,
) -> String {
    let other = 1 - axis;
    let better = match direction {
        ArgReduceDirection::Max => "candidate > best_val",
        ArgReduceDirection::Min => "candidate < best_val",
    };
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> ameta: ArgReduceMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let lane = gid.x;
    if (lane >= ameta.offsets.z) {{
        return;
    }}
    let axis_len = ameta.in_shape[{axis}];
    let lane_base = i32(ameta.offsets.x) + i32(lane) * ameta.in_strides[{other}];
    var best_val: {ty} = input[u32(lane_base)];
    var best_idx: u32 = 0u;
    for (var a: u32 = 1u; a < axis_len; a = a + 1u) {{
        let candidate = input[u32(lane_base + i32(a) * ameta.in_strides[{axis}])];
        if ({better}) {{
            best_val = candidate;
            best_idx = a;
        }}
    }}
    let out_off = i32(ameta.offsets.y) + i32(lane) * ameta.out_strides[{other}];
    out[u32(out_off)] = best_idx;
}}
"#,
        meta = WGSL_ARG_REDUCE_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        axis = axis,
        other = other,
        better = better,
    )
}

/// WGPU device marker implementing [`ArgReduceOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuArgReduceOps;

impl<T> ArgReduceOps<WgpuDevice, T> for WgpuArgReduceOps
where
    T: DialectScalar<Wgsl> + Pod + PartialOrd,
{
    fn argmax_axis_into(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, WgpuBuffer<u32>, 2>,
    ) -> Result<()> {
        arg_reduce_into::<T>(
            device,
            input,
            axis,
            output,
            ArgReduceDirection::Max,
            BlockWidth::DEFAULT,
        )
    }

    fn argmin_axis_into(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, WgpuBuffer<u32>, 2>,
    ) -> Result<()> {
        arg_reduce_into::<T>(
            device,
            input,
            axis,
            output,
            ArgReduceDirection::Min,
            BlockWidth::DEFAULT,
        )
    }
}

fn arg_reduce_into<T>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, 2>,
    axis: usize,
    output: StridedView<'_, WgpuBuffer<u32>, 2>,
    direction: ArgReduceDirection,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    let in_layout = input.layout;
    let out_layout = output.layout;
    let (lanes, _axis_len) =
        validate_arg_reduce_shape(axis, in_layout.shape(), out_layout.shape())?;

    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "arg-reduce output layout must be non-overlapping".to_string(),
        });
    }
    if lanes == 0 {
        return Ok(());
    }

    let in_shape = in_layout.shape();
    let in_strides = in_layout.strides();
    let out_strides = out_layout.strides();
    let meta = ArgReduceMeta {
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
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(lanes, "dispatch size")?,
            0,
        ],
    };

    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<ArgReduceMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<ArgReduceKernel>(),
            TypeId::of::<T>(),
            // `axis` and `direction` fold into the block-width slot of the
            // shared `(TypeId, TypeId, u32)` cache key: distinct axes/
            // directions must never collide on one cached pipeline.
            block_width.get() * 4
                + (axis as u32) * 2
                + matches!(direction, ArgReduceDirection::Min) as u32,
        ),
        "hephaestus-arg-reduce",
        || arg_reduce_shader::<T>(block_width, axis, direction),
    );

    let groups = workgroups(lanes, block_width)?;
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
            label: Some("hephaestus-arg-reduce"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-arg-reduce"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-arg-reduce",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}
