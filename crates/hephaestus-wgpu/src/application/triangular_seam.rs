//! WGPU implementation of [`hephaestus_core::TriangularOps`].
//!
//! One thread per `(row, col)` pair (flat-dispatched over `rows * cols`)
//! evaluates the same `col <=/>= row + diagonal` test
//! [`hephaestus_core::triangular_keeps`] uses, and copies or zeroes
//! directly — no shared state between threads. `mode` bakes into the
//! generated WGSL at pipeline-cache-key granularity (there is no `axis` to
//! bake in — this seam has none); `diagonal` is an ordinary runtime uniform
//! value.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, DialectScalar, HephaestusError, Result, StridedView, TriangularMode, TriangularOps,
    Wgsl, validate_triangular_shape,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::{map_layout_err, to_i32, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `TriangularMeta` uniform. Every field is a full `vec4` for the same
/// reason `ArgReduceMeta` documents (ADR 0063).
///
/// `in_strides`/`out_strides` use lanes `[row, col]`; lanes `[2]`/`[3]` are
/// unused padding. `offsets` is `[in_offset, out_offset, 0, 0]`. `params`
/// is `[rows, cols, diagonal_bits, 0]`, where `diagonal_bits` is
/// `diagonal` bit-cast to `u32` (WGSL reinterprets it back to `i32` in the
/// shader) since the uniform's other lanes are already `u32`-typed.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TriangularMeta {
    in_strides: [i32; 4],
    out_strides: [i32; 4],
    offsets: [u32; 4],
    params: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<TriangularMeta>() == 64);

const WGSL_TRIANGULAR_META: &str = r"struct TriangularMeta {
    in_strides: vec4<i32>,
    out_strides: vec4<i32>,
    offsets: vec4<u32>,
    params: vec4<u32>,
}
";

struct TriangularKernel;

fn triangular_shader<T: DialectScalar<Wgsl>>(width: BlockWidth, mode: TriangularMode) -> String {
    let keep = match mode {
        TriangularMode::Lower => "col <= row + diagonal",
        TriangularMode::Upper => "col >= row + diagonal",
    };
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> ameta: TriangularMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let rows = ameta.params.x;
    let cols = ameta.params.y;
    let diagonal = bitcast<i32>(ameta.params.z);
    let total = rows * cols;
    if (gid.x >= total) {{
        return;
    }}
    let row = i32(gid.x / cols);
    let col = i32(gid.x % cols);

    let out_off = i32(ameta.offsets.y) + row * ameta.out_strides.x + col * ameta.out_strides.y;
    if ({keep}) {{
        let in_off = i32(ameta.offsets.x) + row * ameta.in_strides.x + col * ameta.in_strides.y;
        out[u32(out_off)] = input[u32(in_off)];
    }} else {{
        out[u32(out_off)] = {ty}(0.0);
    }}
}}
"#,
        meta = WGSL_TRIANGULAR_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        keep = keep,
    )
}

/// WGPU device marker implementing [`TriangularOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuTriangularOps;

impl<T> TriangularOps<WgpuDevice, T> for WgpuTriangularOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn triangular_into(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        mode: TriangularMode,
        diagonal: i64,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let out_layout = output.layout;
        validate_triangular_shape(in_layout.shape(), out_layout.shape())?;

        in_layout
            .validate_storage_len(input.buffer.len)
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len)
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "triangular output layout must be non-overlapping".to_string(),
            });
        }
        let [rows, cols] = in_layout.shape();
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let block_width = BlockWidth::DEFAULT;

        let diagonal_i32 =
            i32::try_from(diagonal).map_err(|_| HephaestusError::InvalidConfiguration {
                message: "triangular diagonal exceeds i32 range".to_string(),
            })?;

        let in_strides = in_layout.strides();
        let out_strides = out_layout.strides();
        let meta = TriangularMeta {
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
                0,
                0,
            ],
            params: [
                to_u32(rows, "row count")?,
                to_u32(cols, "column count")?,
                diagonal_i32 as u32,
                0,
            ],
        };

        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<TriangularMeta>(1)?)?;
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let pipeline = cached_pipeline(
            device,
            (
                TypeId::of::<TriangularKernel>(),
                TypeId::of::<T>(),
                block_width.get() * 2 + matches!(mode, TriangularMode::Upper) as u32,
            ),
            "hephaestus-triangular",
            || triangular_shader::<T>(block_width, mode),
        );

        let total =
            rows.checked_mul(cols)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: "triangular dispatch size overflows".to_string(),
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
                label: Some("hephaestus-triangular"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-triangular"),
            });
        encode_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            groups,
            "hephaestus-triangular",
        );
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
