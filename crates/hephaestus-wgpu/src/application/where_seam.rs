//! WGPU implementation of [`hephaestus_core::WhereOps`].
//!
//! One packed-metadata kernel serves every rank up to
//! [`MAX_STRIDED_RANK`](crate::application::strided::MAX_STRIDED_RANK): one
//! thread per output cell decodes its coordinate, reads the condition and
//! both branch cells at that coordinate, and writes the selected branch.
//! The condition test is scalar inequality against zero, never a bitwise
//! test. WGSL does not advertise IEEE special values, so a `NaN`
//! condition selects an unspecified branch on this backend — the shared
//! special-values clause skips that cell here by construction (ADR 0043) —
//! while every finite edge (`-0.0` selects `on_false`, any nonzero
//! magnitude selects `on_true`) holds exactly as on the host.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, StridedView, Wgsl, WhereOps,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{
    MAX_STRIDED_RANK, map_layout_err, pad_shape, pad_strides, to_u32,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `WhereMeta` uniform: rank-8 padded output shape, condition,
/// `on_true`, `on_false`, and output strides, `[cond_offset,
/// true_offset, false_offset, output_offset]`, and `[len, _, _, _]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WhereMeta {
    out_shape: [u32; 8],
    cond_strides: [i32; 8],
    true_strides: [i32; 8],
    false_strides: [i32; 8],
    out_strides: [i32; 8],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<WhereMeta>() == 192);

const WGSL_WHERE_META: &str = r"struct WhereMeta {
    out_shape: array<vec4<u32>, 2>,
    cond_strides: array<vec4<i32>, 2>,
    true_strides: array<vec4<i32>, 2>,
    false_strides: array<vec4<i32>, 2>,
    out_strides: array<vec4<i32>, 2>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

fn where_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> wmeta: WhereMeta;
@group(0) @binding(1) var<storage, read> cond: array<{ty}>;
@group(0) @binding(2) var<storage, read> on_true: array<{ty}>;
@group(0) @binding(3) var<storage, read> on_false: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= wmeta.params.x) {{
        return;
    }}
    var rem = i;
    var cond_off = i32(wmeta.offsets.x);
    var true_off = i32(wmeta.offsets.y);
    var false_off = i32(wmeta.offsets.z);
    var out_off = i32(wmeta.offsets.w);
    for (var p: i32 = 7; p >= 0; p = p - 1) {{
        let group = p / 4;
        let lane = p % 4;
        let out_dim = wmeta.out_shape[group][lane];
        let c = rem % out_dim;
        rem = rem / out_dim;
        cond_off = cond_off + i32(c) * wmeta.cond_strides[group][lane];
        true_off = true_off + i32(c) * wmeta.true_strides[group][lane];
        false_off = false_off + i32(c) * wmeta.false_strides[group][lane];
        out_off = out_off + i32(c) * wmeta.out_strides[group][lane];
    }}
    if (cond[u32(cond_off)] != {ty}(0)) {{
        out[u32(out_off)] = on_true[u32(true_off)];
    }} else {{
        out[u32(out_off)] = on_false[u32(false_off)];
    }}
}}
"#,
        meta = WGSL_WHERE_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

struct WhereKernel;

/// WGPU device marker implementing [`hephaestus_core::WhereOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuWhereOps;

impl<T> WhereOps<WgpuDevice, T> for WgpuWhereOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;

    fn where_into<const N: usize>(
        &self,
        device: &WgpuDevice,
        cond: StridedView<'_, WgpuBuffer<T>, N>,
        on_true: StridedView<'_, WgpuBuffer<T>, N>,
        on_false: StridedView<'_, WgpuBuffer<T>, N>,
        output: StridedView<'_, WgpuBuffer<T>, N>,
    ) -> Result<()> {
        where_into::<T, N>(device, cond, on_true, on_false, output, BlockWidth::DEFAULT)
    }
}

/// Write `on_true` under nonzero condition cells, `on_false` elsewhere,
/// at a caller-selected [`BlockWidth`].
///
/// # Errors
///
/// Returns a typed dispatch error when the four shapes disagree, the
/// output buffer aliases a source buffer, a layout is unsupported, or the
/// backend dispatch fails.
pub fn where_into<T, const N: usize>(
    device: &WgpuDevice,
    cond: StridedView<'_, WgpuBuffer<T>, N>,
    on_true: StridedView<'_, WgpuBuffer<T>, N>,
    on_false: StridedView<'_, WgpuBuffer<T>, N>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    block_width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    const {
        assert!(N <= MAX_STRIDED_RANK, "where dispatch supports rank <= 8");
    }

    hephaestus_core::validate_where_shape(
        cond.layout.shape(),
        on_true.layout.shape(),
        on_false.layout.shape(),
        output.layout.shape(),
    )?;

    let cond_layout = cond.layout;
    let true_layout = on_true.layout;
    let false_layout = on_false.layout;
    let out_layout = output.layout;
    cond_layout
        .validate_storage_len(cond.buffer.len)
        .map_err(map_layout_err)?;
    true_layout
        .validate_storage_len(on_true.buffer.len)
        .map_err(map_layout_err)?;
    false_layout
        .validate_storage_len(on_false.buffer.len)
        .map_err(map_layout_err)?;
    out_layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "where output layout must be non-overlapping".to_string(),
        });
    }
    if output.buffer.aliases(cond.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "where output buffer must not alias the cond buffer".to_string(),
        });
    }
    if output.buffer.aliases(on_true.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "where output buffer must not alias the on_true buffer".to_string(),
        });
    }
    if output.buffer.aliases(on_false.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "where output buffer must not alias the on_false buffer".to_string(),
        });
    }
    let len = out_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(());
    }

    let meta = WhereMeta {
        out_shape: pad_shape(out_layout.shape())?,
        cond_strides: pad_strides(cond_layout.strides())?,
        true_strides: pad_strides(true_layout.strides())?,
        false_strides: pad_strides(false_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            to_u32(cond_layout.offset(), "cond offset")?,
            to_u32(true_layout.offset(), "on_true offset")?,
            to_u32(false_layout.offset(), "on_false offset")?,
            to_u32(out_layout.offset(), "output offset")?,
        ],
        params: [to_u32(len, "dispatch size")?, 0, 0, 0],
    };
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<WhereMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<WhereKernel>(),
            TypeId::of::<T>(),
            block_width.get(),
        ),
        "hephaestus-where",
        || where_shader::<T>(block_width),
    );

    let groups = workgroups(len, block_width)?;

    let mut entries = BindGroupEntries::with_capacity(5);
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: meta_buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 1,
        resource: cond.buffer.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 2,
        resource: on_true.buffer.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: on_false.buffer.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: 4,
        resource: output.buffer.buffer.as_entire_binding(),
    });
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-where"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-where"),
        });
    encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        "hephaestus-where",
    );
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn where_meta_matches_the_wgsl_uniform_layout() {
        assert_eq!(core::mem::size_of::<WhereMeta>(), 192);
    }
}
