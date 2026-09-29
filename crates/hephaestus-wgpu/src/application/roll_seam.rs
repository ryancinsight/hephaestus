//! WGPU implementation of [`hephaestus_core::RollOps`].
//!
//! One thread per `(lane, dst_idx)` pair (flat-dispatched over
//! `lanes * axis_len`) computes its own wrapped source index and copies
//! directly — no shared state between threads. `axis` bakes into the
//! generated WGSL at pipeline-cache-key granularity; `shift` is an ordinary
//! runtime uniform value (it does not change which shader instructions are
//! legal, only the wrap arithmetic's operand).

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, RollOps, StridedView, Wgsl,
    validate_roll_shape,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{encode_compute_pass, try_cached_pipeline, workgroups};
use crate::application::strided::{map_layout_err, to_i32, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `RollMeta` uniform. Every field is a full `vec4` for the same
/// reason `ArgReduceMeta` documents (ADR 0063).
///
/// `in_strides`/`out_strides` use lanes `[axis, other]`; lanes `[2]`/`[3]`
/// are unused padding. `offsets` is `[in_offset, out_offset, 0, 0]`.
/// `params` is `[axis_len, lanes, shift_rem, 0]`, where `shift_rem` is
/// `shift.rem_euclid(axis_len)` precomputed host-side so the shader never
/// needs a signed modulo on a value that could still be negative after one
/// reduction (`rem_euclid` on `i64` narrowed to `u32` here is always
/// non-negative and less than `axis_len` by construction).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RollMeta {
    in_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<RollMeta>() == 64);

const WGSL_ROLL_META: &str = r"struct RollMeta {
    in_strides: vec4<i32>,
    out_strides: vec4<i32>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

struct RollKernel;

fn validate_shader_addressing(layout: &leto::Layout<2>, label: &str) -> Result<()> {
    let mut minimum = layout.offset() as i128;
    let mut maximum = minimum;
    for axis in 0..2 {
        let extent = layout.shape()[axis].saturating_sub(1) as i128;
        let delta = (layout.strides()[axis] as i128)
            .checked_mul(extent)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("{label} stride product overflows shader address arithmetic"),
            })?;
        if delta < i128::from(i32::MIN) || delta > i128::from(i32::MAX) {
            return Err(HephaestusError::DispatchFailed {
                message: format!("{label} stride product exceeds signed shader address range"),
            });
        }
        if delta < 0 {
            minimum =
                minimum
                    .checked_add(delta)
                    .ok_or_else(|| HephaestusError::DispatchFailed {
                        message: format!("{label} minimum address overflows shader arithmetic"),
                    })?;
        } else {
            maximum =
                maximum
                    .checked_add(delta)
                    .ok_or_else(|| HephaestusError::DispatchFailed {
                        message: format!("{label} maximum address overflows shader arithmetic"),
                    })?;
        }
    }
    if minimum < 0 || maximum > i128::from(i32::MAX) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{label} address exceeds signed shader range"),
        });
    }
    Ok(())
}

fn validate_workgroup_limit(groups: u32, limit: u32) -> Result<()> {
    if groups > limit {
        return Err(HephaestusError::DispatchFailed {
            message: format!("roll dispatch requires {groups} workgroups, device limit is {limit}"),
        });
    }
    Ok(())
}

fn roll_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> ameta: RollMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let axis_len = ameta.params.x;
    let lanes = ameta.params.y;
    let shift_rem = ameta.params.z;
    let total = lanes * axis_len;
    if (gid.x >= total) {{
        return;
    }}
    let lane = gid.x / axis_len;
    let dst_idx = gid.x % axis_len;

    let src_idx = (dst_idx + axis_len - shift_rem) % axis_len;

    let in_base = i32(ameta.offsets.x) + i32(lane) * ameta.in_strides.y;
    let in_off = in_base + i32(src_idx) * ameta.in_strides.x;
    let out_base = i32(ameta.offsets.y) + i32(lane) * ameta.out_strides.y;
    let out_off = out_base + i32(dst_idx) * ameta.out_strides.x;
    out[u32(out_off)] = input[u32(in_off)];
}}
"#,
        meta = WGSL_ROLL_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

/// WGPU device marker implementing [`RollOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuRollOps;

impl<T> RollOps<WgpuDevice, T> for WgpuRollOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn roll_axis_into(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        axis: usize,
        shift: i64,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let out_layout = output.layout;
        let (lanes, axis_len) = validate_roll_shape(axis, in_layout.shape(), out_layout.shape())?;

        in_layout
            .validate_storage_len(input.buffer.len)
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len)
            .map_err(map_layout_err)?;
        validate_shader_addressing(in_layout, "roll input")?;
        validate_shader_addressing(out_layout, "roll output")?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "roll output layout must be non-overlapping".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "roll output buffer must not alias input buffer".to_string(),
            });
        }
        if lanes == 0 {
            return Ok(());
        }
        let block_width = BlockWidth::DEFAULT;
        let other = 1 - axis;

        let axis_len_i64 =
            i64::try_from(axis_len).map_err(|_| HephaestusError::InvalidConfiguration {
                message: "roll axis length exceeds i64 range".to_string(),
            })?;
        let shift_rem = shift.rem_euclid(axis_len_i64);

        let total =
            lanes
                .checked_mul(axis_len)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: "roll dispatch size overflows".to_string(),
                })?;
        let groups = workgroups(total, block_width)?;
        let limit = device.limits().max_compute_workgroups_per_dimension;
        validate_workgroup_limit(groups, limit)?;

        let in_strides = in_layout.strides();
        let out_strides = out_layout.strides();
        let meta = RollMeta {
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
                to_u32(axis_len, "axis length")?,
                to_u32(lanes, "dispatch size")?,
                to_u32(
                    usize::try_from(shift_rem).map_err(|_| {
                        HephaestusError::InvalidConfiguration {
                            message: "roll shift remainder exceeds usize range".to_string(),
                        }
                    })?,
                    "shift remainder",
                )?,
                0,
            ],
        };

        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<RollMeta>(1)?)?;
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let pipeline = try_cached_pipeline(
            device,
            (
                TypeId::of::<RollKernel>(),
                TypeId::of::<T>(),
                block_width.get(),
            ),
            "hephaestus-roll",
            || roll_shader::<T>(block_width),
        )?;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Layout;

    #[test]
    fn rejects_workgroups_above_device_limit() {
        let error = validate_workgroup_limit(9, 8).expect_err("dispatch limit");
        assert!(matches!(error, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn rejects_shader_address_overflow() {
        let layout = Layout::try_new([2, 2], [i32::MAX as isize, 1], 0).expect("layout");
        let error = validate_shader_addressing(&layout, "input").expect_err("address range");
        assert!(matches!(error, HephaestusError::DispatchFailed { .. }));
    }
}
