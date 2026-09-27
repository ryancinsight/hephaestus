//! WGPU implementation of [`hephaestus_core::PadOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK), mirroring
//! [`crate::application::strided`]'s decode/encode core: the output's logical flat index
//! decodes into a per-axis coordinate, which maps to an input coordinate by
//! subtracting that axis's `before` pad width. An input coordinate that
//! underflows (still `>= 0` conceptually, but the subtraction wraps past
//! `u32::MAX` for an out-of-range unsigned width) or reaches the input's
//! extent selects the fill value instead of a source read — one bounds check
//! covers both pad margins without a branch per margin per axis.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, PadOps, PadWidth, Result,
    StridedView, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `PadMeta` uniform: rank-8 padded output/input shapes, input and
/// output strides, per-axis pad-before widths, and
/// `[input_offset, output_offset, len, _pad]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PadMeta {
    out_shape: [u32; 8],
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    pad_before: [u32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<PadMeta>() == 176);

const WGSL_PAD_META: &str = r"struct PadMeta {
    out_shape: array<vec4<u32>, 2>,
    in_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    pad_before: array<vec4<u32>, 2>,
    offsets: vec4<u32>,
}
";

fn pad_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> pmeta: PadMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<uniform> fill: {ty};
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= pmeta.offsets.z) {{
        return;
    }}
    var rem = i;
    var in_off = i32(pmeta.offsets.x);
    var out_off = i32(pmeta.offsets.y);
    var in_bounds = true;
    for (var d: i32 = 7; d >= 0; d = d - 1) {{
        let group = d / 4;
        let lane = d % 4;
        let out_dim = pmeta.out_shape[group][lane];
        let out_idx = rem % out_dim;
        rem = rem / out_dim;
        out_off = out_off + i32(out_idx) * pmeta.out_strides[group][lane];
        let before = pmeta.pad_before[group][lane];
        let in_idx = out_idx - before;
        let in_dim = pmeta.in_shape[group][lane];
        if (in_idx >= in_dim) {{
            in_bounds = false;
        }}
        in_off = in_off + i32(in_idx) * pmeta.in_strides[group][lane];
    }}
    if (in_bounds) {{
        out[u32(out_off)] = input[u32(in_off)];
    }} else {{
        out[u32(out_off)] = fill;
    }}
}}
"#,
        meta = WGSL_PAD_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct PadKernel;

/// WGPU device marker implementing [`hephaestus_core::PadOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuPadOps;

impl<T> PadOps<WgpuDevice, T> for WgpuPadOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn pad_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        width: PadWidth<N>,
        fill: T,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        pad_into::<T, N>(device, input, width, fill, output, BlockWidth::DEFAULT)
    }
}

/// Write a padded copy of `input` into `output` at a caller-selected
/// [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `output`'s shape does not match the
/// padded shape implied by `width`, a layout is unsupported, or the backend
/// dispatch fails.
pub fn pad_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    width: PadWidth<N>,
    fill: T,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "pad dispatch supports rank <= 8");
    }

    let in_layout = input.layout;
    let out_layout = output.layout;

    hephaestus_core::validate_pad_shape(in_layout.shape(), width, out_layout.shape())?;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "pad output layout must be non-overlapping".to_string(),
        });
    }
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(());
    }

    let mut pad_before = [0u32; 8];
    for d in 0..N {
        pad_before[8 - N + d] = to_u32(width[d].0, "pad before-width")?;
    }

    let meta = PadMeta {
        out_shape: pad_shape(out_layout.shape())?,
        in_shape: pad_shape(in_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        pad_before,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
            0,
        ],
    };

    let fill_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<T>(1)?)?;
    device
        .queue()
        .write_buffer(&fill_buffer, 0, eunomia::layout::bytes_of(&fill));

    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<PadMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<PadKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-pad",
        || pad_shader::<T>(block_width),
    );

    let groups = workgroups(len, block_width)?;

    // A zero-length input cannot bind as a storage buffer: wgpu enforces a
    // nonzero minimum binding size, but every output cell is provably `fill`
    // whenever any input axis is empty (`in_shape[d] == 0` makes every
    // `in_idx >= in_dim` in the kernel), so the buffer is never read. A tiny
    // throwaway allocation stands in for the binding in that case only.
    let empty_input_placeholder;
    let input_binding: &wgpu::Buffer = if input.buffer.len == 0 {
        empty_input_placeholder = device.alloc_uninitialized::<T>(1)?;
        &empty_input_placeholder.buffer
    } else {
        &input.buffer.buffer
    };

    let mut entries = BindGroupEntries::with_capacity(4);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: meta_buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 1,
        resource: input_binding.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 2,
        resource: fill_buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: output.buffer.buffer.as_entire_binding(),
    });
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-pad"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-pad"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-pad",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<PadMeta>(), 176);
    }
}
