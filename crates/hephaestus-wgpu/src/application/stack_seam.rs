//! WGPU implementation of [`hephaestus_core::StackOps`].
//!
//! One packed-metadata kernel serves every input rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK) minus
//! one: each input dispatches one block-copy pass over its own cells, and
//! all passes share a single pipeline inside a single encoder. A thread
//! decodes its input coordinate from the input shape, reads through the
//! input strides, and writes the same coordinate with the new `axis` lane
//! set to the input's index — no bounds check is needed because validation
//! proves each output window equals its input shape exactly. WebGPU has no
//! portable multi-buffer binding for a runtime input count (binding arrays
//! need non-uniform-indexing features no other seam requires), so per-input
//! passes are the portable strategy; a fused single-launch kernel stays
//! future work for a profiled need, as the stack ADR records.
//!
//! The decode loop walks padded *input* lanes while the output has one more
//! lane than the input, so the shader maps each input lane to its output
//! lane arithmetically: input lane `d` lands on output lane `d` below the
//! new axis and `d + 1` at or above it, and the new-axis lane itself is
//! seeded once from the input index before the loop. Padding lanes are
//! skipped: their index is always zero, and mapping one would read an
//! out-of-range stride lane.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, StackOps, StridedView, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `StackMeta` uniform: rank-8 padded input shape, input and output
/// strides, `[input_offset, output_offset, len, axis]`, and
/// `[input_index, input_pad, _, _]`.
///
/// `axis` is the logical new-axis position (`0..=N`); `input_pad` is `8 - N`,
/// the first padded lane holding a real input dimension.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StackMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<StackMeta>() == 128);

const WGSL_STACK_META: &str = r"struct StackMeta {
    in_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn stack_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> cmeta: StackMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= cmeta.offsets.z) {{
        return;
    }}
    let axis = i32(cmeta.offsets.w);
    let in_pad = i32(cmeta.params.y);
    // The padded lane of the new axis: output lanes start at `in_pad - 1`
    // because the output rank is one more than the input rank.
    let q_axis = in_pad + axis - 1;
    var rem = i;
    var in_off = i32(cmeta.offsets.x);
    var out_off = i32(cmeta.offsets.y)
        + i32(cmeta.params.x) * cmeta.out_strides[q_axis / 4][q_axis % 4];
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let in_dim = cmeta.in_shape[group][lane];
        let in_idx = rem % in_dim;
        rem = rem / in_dim;
        in_off = in_off + i32(in_idx) * cmeta.in_strides[group][lane];
        if (p >= in_pad) {{
            let d_in = p - in_pad;
            var d_out = d_in;
            if (d_in >= axis) {{
                d_out = d_out + 1;
            }}
            let q = in_pad - 1 + d_out;
            out_off = out_off + i32(in_idx) * cmeta.out_strides[q / 4][q % 4];
        }}
    }}
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_STACK_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct StackKernel;

/// WGPU device marker implementing [`hephaestus_core::StackOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuStackOps;

impl<T> StackOps<WgpuDevice, T> for WgpuStackOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn stack_into<const N: usize, const M: usize>(
        &self,
        device: &WgpuDevice,
        inputs: &[StridedView<'_, WgpuBuffer<T>, N>],
        axis: usize,
        output: StridedView<'_, WgpuBuffer<T>, M>,
    ) -> Result<()>
    where
        leto::RankMarker<N>: leto::InsertAxis<N, LargerShape = [usize; M]>,
    {
        stack_into::<T, N, M>(device, inputs, axis, output, BlockWidth::DEFAULT)
    }
}

/// Write the stack of `inputs` along the new `axis` into `output` at a
/// caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `M != N + 1`, `axis` exceeds `N`,
/// `inputs` is empty, the inputs disagree on any dimension, `output`'s
/// shape does not match the stacked shape, the output buffer aliases an
/// input buffer, a layout is unsupported, or the backend dispatch fails.
pub fn stack_into<T, const N: usize, const M: usize>(
    device: &WgpuDevice,
    inputs: &[StridedView<'_, WgpuBuffer<T>, N>],
    axis: usize,
    output: StridedView<'_, WgpuBuffer<T>, M>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(
            N < MAX_STRIDED_RANK,
            "stack dispatch supports input rank <= 7"
        );
    }

    let shapes: Vec<[usize; N]> = inputs.iter().map(|view| view.layout.shape()).collect();
    hephaestus_core::validate_stack_shape(&shapes, axis, output.layout.shape())?;

    let out_layout = output.layout;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "stack output layout must be non-overlapping".to_string(),
        });
    }
    if inputs
        .iter()
        .any(|input| output.buffer.aliases(input.buffer))
    {
        return Err(HephaestusError::DispatchFailed {
            message: "stack output buffer must not alias any input buffer".to_string(),
        });
    }
    if out_layout.checked_size().map_err(map_layout_err)? == 0 {
        return Ok(());
    }

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<StackKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-stack",
        || stack_shader::<T>(block_width),
    );

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-stack"),
        });

    let logical_axis = to_u32(axis, "stack axis")?;
    let input_pad = to_u32(8 - N, "stack input pad")?;
    let out_strides = pad_strides(out_layout.strides())?;
    let out_offset = to_u32(out_layout.offset(), "output offset")?;

    // Every pass keeps its uniform buffer alive until the shared submit:
    // dropping a pooled buffer returns it to the recycle pool, so rebinding
    // one handle per pass would hand every bind group the same buffer with
    // the last pass's metadata. Holding them here forces distinct buffers,
    // exactly as the cat seam holds its per-input buffers past dispatch.
    let mut meta_buffers = Vec::with_capacity(inputs.len());

    for (index, input) in inputs.iter().enumerate() {
        let in_layout = input.layout;
        in_layout
            .validate_storage_len(input.buffer.len)
            .map_err(map_layout_err)?;
        let len = in_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            // An empty input contributes no cells; skipping its pass also
            // sidesteps the nonzero minimum binding size for empty buffers.
            continue;
        }

        let meta = StackMeta {
            in_shape: pad_shape(in_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            out_strides,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                out_offset,
                to_u32(len, "dispatch size")?,
                logical_axis,
            ],
            params: [to_u32(index, "input index")?, input_pad, 0, 0],
        };
        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<StackMeta>(1)?)?;
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let groups = workgroups(len, block_width)?;

        // The entries drop before the uniform buffer moves into the
        // held set: the entry vector's `Drop` would otherwise extend its
        // borrow of the buffer past the move.
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
                    label: Some("hephaestus-stack"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                })
        };

        encode_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            groups,
            "hephaestus-stack",
        );
        meta_buffers.push(meta_buffer);
    }
    device.queue().submit(Some(encoder.finish()));
    drop(meta_buffers);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<StackMeta>(), 128);
    }
}
