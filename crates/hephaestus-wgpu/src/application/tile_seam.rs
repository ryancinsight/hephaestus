//! WGPU implementation of [`hephaestus_core::TileOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK): one
//! thread per output cell decodes its coordinate, takes each lane modulo
//! the input extent, and copies the addressed input cell. The rep counts
//! never reach the kernel: validation proves each output extent is its
//! input extent times its reps, so the modulo alone reproduces the tiling,
//! and a zero rep count empties its axis before any thread runs.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, StridedView, TileOps, TileReps, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `TileMeta` uniform: rank-8 padded output and input shapes, input
/// and output strides, and `[input_offset, output_offset, len, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileMeta {
    out_shape: [u32; 8],
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<TileMeta>() == 144);

const WGSL_TILE_META: &str = r"struct TileMeta {
    out_shape: array<vec4<u32>, 2>,
    in_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
}
";

fn tile_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> tmeta: TileMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= tmeta.offsets.z) {{
        return;
    }}
    var rem = i;
    var in_off = i32(tmeta.offsets.x);
    var out_off = i32(tmeta.offsets.y);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = tmeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        // Every live lane has a nonzero input extent: `out = in * reps`
        // with `out > 0` proves `in > 0`, and padded lanes hold ones.
        let src_c = c % tmeta.in_shape[group][lane];
        in_off = in_off + i32(src_c) * tmeta.in_strides[group][lane];
        out_off = out_off + i32(c) * tmeta.out_strides[group][lane];
    }}
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_TILE_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct TileKernel;

/// WGPU device marker implementing [`hephaestus_core::TileOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuTileOps;

impl<T> TileOps<WgpuDevice, T> for WgpuTileOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn tile_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        reps: TileReps<N>,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        tile_into::<T, N>(device, input, reps, output, BlockWidth::DEFAULT)
    }
}

/// Write `reps[d]` concatenated copies of `input` along every axis at a
/// caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when an output extent is not its input
/// extent times its rep count, a rep count overflows, the output buffer
/// aliases the input buffer, a layout is unsupported, or the backend
/// dispatch fails.
pub fn tile_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    reps: TileReps<N>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "tile dispatch supports rank <= 8");
    }

    hephaestus_core::validate_tile_shape(input.layout.shape(), reps, output.layout.shape())?;

    let in_layout = input.layout;
    let out_layout = output.layout;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "tile output layout must be non-overlapping".to_string(),
        });
    }
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "tile output buffer must not alias the input buffer".to_string(),
        });
    }
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(());
    }

    let meta = TileMeta {
        out_shape: pad_shape(out_layout.shape())?,
        in_shape: pad_shape(in_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
            0,
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<TileMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<TileKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-tile",
        || tile_shader::<T>(block_width),
    );

    let groups = workgroups(len, block_width)?;

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
            label: Some("hephaestus-tile"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-tile"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-tile",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<TileMeta>(), 144);
    }
}
