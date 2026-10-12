//! WGPU implementation of [`hephaestus_core::RollOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK): one
//! thread per output cell decodes its coordinate, moves each rolled lane
//! back by its effective shift modulo the extent, and copies the addressed
//! input cell. Rolls along distinct dims commute and rolls along one dim
//! add, so the host folds every `(shift, dim)` pair into one effective
//! per-axis displacement before dispatch — duplicate dims compose by
//! summation, and arbitrarily large or negative shifts reduce once, in
//! exact integer arithmetic, rather than per thread.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, RollOps, StridedView, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `RollMeta` uniform: rank-8 padded output shape, input and output
/// strides, per-axis effective back-shifts, and `[input_offset,
/// output_offset, len, _]`.
///
/// Each back-shift is already reduced modulo its dim extent, so the kernel
/// subtracts without underflow; unrolled lanes hold zero. Shapes pack
/// leading ones, so the operation rank occupies the trailing lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RollMeta {
    out_shape: [u32; 8],
    in_strides: [i32; 8],
    out_strides: [i32; 8],
    back_shifts: [u32; 8],
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<RollMeta>() == 144);

const WGSL_ROLL_META: &str = r"struct RollMeta {
    out_shape: array<vec4<u32>, 2>,
    in_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    back_shifts: array<vec4<u32>, 2>,
    offsets: vec4<u32>,
}
";

fn roll_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> rmeta: RollMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= rmeta.offsets.z) {{
        return;
    }}
    var rem = i;
    var in_off = i32(rmeta.offsets.x);
    var out_off = i32(rmeta.offsets.y);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = rmeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        // Input and output share shape, so the output extent is the roll
        // modulus; the back-shift is already reduced below it.
        let src_c = (c + out_dim - rmeta.back_shifts[group][lane]) % out_dim;
        in_off = in_off + i32(src_c) * rmeta.in_strides[group][lane];
        out_off = out_off + i32(c) * rmeta.out_strides[group][lane];
    }}
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_ROLL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct RollKernel;

/// WGPU device marker implementing [`hephaestus_core::RollOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuRollOps;

impl<T> RollOps<WgpuDevice, T> for WgpuRollOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn roll_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        shifts: &[isize],
        dims: &[usize],
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        roll_into::<T, N>(device, input, shifts, dims, output, BlockWidth::DEFAULT)
    }
}

/// Write `input` circularly shifted by `shifts[k]` along `dims[k]` at a
/// caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when `shifts` and `dims` disagree in
/// length, a dim is out of range, the output shape is not the input shape,
/// the output buffer aliases the input buffer, a layout is unsupported,
/// or the backend dispatch fails.
pub fn roll_into<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
    shifts: &[isize],
    dims: &[usize],
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "roll dispatch supports rank <= 8");
    }

    hephaestus_core::validate_roll_shape(
        input.layout.shape(),
        shifts,
        dims,
        output.layout.shape(),
    )?;

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
            message: "roll output layout must be non-overlapping".to_string(),
        });
    }
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "roll output buffer must not alias the input buffer".to_string(),
        });
    }
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        // Every extent is nonzero below this point, so no back-shift
        // normalizes against a zero modulus.
        return Ok(());
    }

    // Fold every pair into one effective per-axis shift in i128, then
    // reduce modulo the extent: exact, once per axis, never per thread.
    let mut totals = [0i128; N];
    for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
        totals[dim] += shift as i128;
    }
    let mut back_shifts = [0u32; 8];
    for (d, &total) in totals.iter().enumerate() {
        let extent = in_layout.shape()[d];
        // The reduced shift is below the extent, hence below `usize::MAX`;
        // the fallible conversion keeps the function total regardless.
        let reduced = total.rem_euclid(extent as i128);
        let reduced_usize =
            usize::try_from(reduced).map_err(|_| HephaestusError::InvalidConfiguration {
                message: format!("roll back-shift {reduced} exceeds the usize range"),
            })?;
        back_shifts[8 - N + d] = to_u32(reduced_usize, "roll back-shift")?;
    }

    let meta = RollMeta {
        out_shape: pad_shape(out_layout.shape())?,
        in_strides: pad_strides(in_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        back_shifts,
        offsets: [
            to_u32(in_layout.offset(), "input offset")?,
            to_u32(out_layout.offset(), "output offset")?,
            to_u32(len, "dispatch size")?,
            0,
        ],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<RollMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<RollKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-roll",
        || roll_shader::<T>(block_width),
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
            label: Some("hephaestus-roll"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-roll"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-roll",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roll_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<RollMeta>(), 144);
    }
}
