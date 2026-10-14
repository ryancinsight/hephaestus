//! WGPU implementation of [`hephaestus_core::MaskedFillOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK): one
//! thread per output cell decodes its coordinate, reads the input and mask
//! cells at that coordinate, and writes the fill value or the input cell
//! depending on the mask test. The mask test is scalar inequality against
//! zero, never a bitwise test: WGSL follows IEEE equality, so a `NaN`
//! mask fills (unordered compares unequal) while a negative-zero mask
//! keeps the input, exactly as the host reference does. The fill value
//! crosses as a second uniform buffer, the way the pad seam's fill does.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, MaskedFillOps, Result, StridedView, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `MaskedFillMeta` uniform: rank-8 padded output shape, input, mask,
/// and output strides, and `[input_offset, mask_offset, output_offset,
/// len]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskedFillMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    mask_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<MaskedFillMeta>() == 144);

const WGSL_MASKED_FILL_META: &str = r"struct MaskedFillMeta {
    out_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    mask_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
}
";

fn masked_fill_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> mmeta: MaskedFillMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read> mask: array<{ty}>;
@group(0) @binding(3) var<uniform> value: {ty};
@group(0) @binding(4) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= mmeta.offsets.w) {{
        return;
    }}
    var rem = i;
    var in_off = i32(mmeta.offsets.x);
    var mask_off = i32(mmeta.offsets.y);
    var out_off = i32(mmeta.offsets.z);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = mmeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        in_off = in_off + i32(c) * mmeta.in_strides[group][lane];
        mask_off = mask_off + i32(c) * mmeta.mask_strides[group][lane];
        out_off = out_off + i32(c) * mmeta.out_strides[group][lane];
    }}
    if (mask[u32(mask_off)] != {ty}(0)) {{
        out[u32(out_off)] = value;
    }} else {{
        out[u32(out_off)] = input[u32(in_off)];
    }}
}}
"#,
        meta = WGSL_MASKED_FILL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct MaskedFillKernel;

/// WGPU device marker implementing [`hephaestus_core::MaskedFillOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuMaskedFillOps;

impl<T> MaskedFillOps<WgpuDevice, T> for WgpuMaskedFillOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;

    fn masked_fill_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        mask: StridedView<'_, WgpuBuffer<T>, N>,
        value: T,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        masked_fill_into::<T, N>(device, input, mask, value, output, BlockWidth::DEFAULT)
    }
}

/// Write `value` under nonzero mask cells, copying `input` elsewhere, at
/// a caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when the three shapes disagree, the
/// output buffer aliases the input or mask buffer, a layout is
/// unsupported, or the backend dispatch fails.
pub fn masked_fill_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    mask: StridedView<'_, WgpuBuffer<T>, N>,
    value: T,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(
            N <= MAX_STRIDED_RANK,
            "masked_fill dispatch supports rank <= 8"
        );
    }

    hephaestus_core::validate_masked_fill_shape(
        input.layout.shape(),
        mask.layout.shape(),
        output.layout.shape(),
    )?;

    let in_layout = input.layout;
    let mask_layout = mask.layout;
    let out_layout = output.layout;
    in_layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    mask_layout
        .validate_storage_len(mask.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "masked_fill output layout must be non-overlapping".to_string(),
        });
    }
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "masked_fill output buffer must not alias the input buffer".to_string(),
        });
    }
    if output.buffer.aliases(mask.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "masked_fill output buffer must not alias the mask buffer".to_string(),
        });
    }
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        // All three shapes agree, so every buffer is empty together: no
        // stand-in binding is needed, unlike the pad seam's empty-input
        // case.
        return Ok(());
    }

    let meta = MaskedFillMeta {
        out_shape: pad_shape(out_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        mask_strides: pad_strides(mask_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(mask_layout.offset(), "mask offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
        ],
    };

    let value_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<T>(1)?)?;
    device
        .queue()
        .write_buffer(&value_buffer, 0, eunomia::layout::bytes_of(&value));

    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<MaskedFillMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<MaskedFillKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-masked_fill",
        || masked_fill_shader::<T>(block_width),
    );

    let groups = workgroups(len, block_width)?;

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
        resource: mask.buffer.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: value_buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 4,
        resource: output.buffer.buffer.as_entire_binding(),
    });
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-masked_fill"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-masked_fill"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-masked_fill",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_fill_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<MaskedFillMeta>(), 144);
    }
}
