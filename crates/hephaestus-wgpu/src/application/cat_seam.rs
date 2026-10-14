//! WGPU implementation of [`hephaestus_core::CatOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK) and every
//! input count: each input dispatches one block-copy pass over its own cells,
//! and all passes share a single pipeline inside a single encoder. A thread
//! decodes its input coordinate from the input shape, reads through the input
//! strides, and writes the same coordinate with the `axis` lane shifted by the
//! input's validated base offset — no bounds check is needed because validation
//! proves the output window equals the input shape exactly. WebGPU has no
//! portable multi-buffer binding for a runtime input count (binding arrays need
//! non-uniform-indexing features no other seam requires), so per-input passes
//! are the portable strategy; a fused single-launch kernel stays future work
//! for a profiled need, as ADR 0071 records.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CatOps, DialectScalar, HephaestusError, Result, StridedView, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `CatMeta` uniform: rank-8 padded input shape, input and output
/// strides, `[input_offset, output_offset, len, axis]`, and
/// `[axis_base, _, _, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CatMeta {
    in_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<CatMeta>() == 128);

const WGSL_CAT_META: &str = r"struct CatMeta {
    in_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn cat_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> cmeta: CatMeta;
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
    let axis = i32(cmeta.offsets.w);
    let base = i32(cmeta.params.x);
    for (var d: i32 = 7; d >= 0; d = d - 1) {{
        let group = d / 4;
        let lane = d % 4;
        let in_dim = cmeta.in_shape[group][lane];
        let in_idx = rem % in_dim;
        rem = rem / in_dim;
        in_off = in_off + i32(in_idx) * cmeta.in_strides[group][lane];
        var out_idx = in_idx;
        if (d == axis) {{
            out_idx = out_idx + u32(base);
        }}
        out_off = out_off + i32(out_idx) * cmeta.out_strides[group][lane];
    }}
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_CAT_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct CatKernel;

/// WGPU device marker implementing [`hephaestus_core::CatOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCatOps;

impl<T> CatOps<WgpuDevice, T> for WgpuCatOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn cat_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        inputs: &[StridedView<'_, WgpuBuffer<T>, N>],
        axis: usize,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        cat_into::<T, N>(device, inputs, axis, output, BlockWidth::DEFAULT)
    }
}

/// Write the concatenation of `inputs` along `axis` into `output` at a
/// caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `inputs` is empty, `axis` is out of
/// range, the inputs disagree on a non-`axis` dimension, `output`'s shape
/// does not match the concatenated shape, a layout is unsupported, or the
/// backend dispatch fails.
pub fn cat_into<T, const N: usize>(
    device: &WgpuDevice,
    inputs: &[StridedView<'_, WgpuBuffer<T>, N>],
    axis: usize,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "cat dispatch supports rank <= 8");
    }

    let shapes: Vec<[usize; N]> = inputs.iter().map(|view| view.layout.shape()).collect();
    let bases = hephaestus_core::validate_cat_shape(&shapes, axis, output.layout.shape())?;

    let out_layout = output.layout;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "cat output layout must be non-overlapping".to_string(),
        });
    }
    if inputs
        .iter()
        .any(|input| output.buffer.aliases(input.buffer))
    {
        return Err(HephaestusError::DispatchFailed {
            message: "cat output buffer must not alias any input buffer".to_string(),
        });
    }
    if out_layout.checked_size().map_err(map_layout_err)? == 0 {
        return Ok(());
    }

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<CatKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-cat",
        || cat_shader::<T>(block_width),
    );

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-cat"),
        });

    // The rank-8 lane holding `axis`: shapes pack leading ones, so the
    // operation rank occupies the trailing lanes.
    let padded_axis = to_u32(8 - N + axis, "cat axis lane")?;
    let out_strides = pad_strides(out_layout.strides())?;
    let out_offset = to_u32(out_layout.offset(), "output offset")?;

    // Every pass keeps its uniform buffer alive until the shared submit:
    // dropping a pooled buffer returns it to the recycle pool, so rebinding
    // one handle per pass would hand every bind group the same buffer with
    // the last pass's metadata. Holding them here forces distinct buffers,
    // exactly as the pad seam holds its fill buffer past dispatch.
    let mut meta_buffers = Vec::with_capacity(inputs.len());

    for (input, base) in inputs.iter().zip(bases.iter()) {
        let in_layout = input.layout;
        in_layout
            .validate_storage_len(input.buffer.len)
            .map_err(map_layout_err)?;
        let len = in_layout.checked_size().map_err(map_layout_err)?;
        if len == 0 {
            // An empty segment contributes no cells; skipping its pass also
            // sidesteps the nonzero minimum binding size for empty buffers.
            continue;
        }

        let meta = CatMeta {
            in_shape: pad_shape(in_layout.shape())?,
            in_strides: pad_strides(in_layout.strides())?,
            out_strides,
            offsets: [
                to_u32(in_layout.offset(), "input offset")?,
                out_offset,
                to_u32(len, "dispatch size")?,
                padded_axis,
            ],
            params: [to_u32(*base, "axis base")?, 0, 0, 0],
        };
        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<CatMeta>(1)?)?;
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
                    label: Some("hephaestus-cat"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                })
        };

        encode_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            groups,
            "hephaestus-cat",
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
    fn cat_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<CatMeta>(), 128);
    }
}
