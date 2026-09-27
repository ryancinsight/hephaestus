//! WGPU implementation of [`hephaestus_core::CrossProductOps`].
//!
//! One thread per `(x, y, z)` triple; no strided metadata is needed since
//! `DenseVectorOps`'s own convention (flat, contiguous buffers) applies here
//! too — the operands are plain `array<T>` bindings and the uniform carries
//! only the triple count.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, CrossProductOps, DeviceBuffer, DialectScalar, HephaestusError, Result, Wgsl,
    validate_cross_product_lengths,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::to_u32;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `CrossMeta` uniform: `[triple_count, _pad, _pad, _pad]`. A full
/// `vec4` even for one value: WGSL's uniform address space still requires
/// the struct's only member aligned to its own size, and matching every
/// other kernel's convention here means a second field never silently
/// desynchronizes the layout later (see ADR 0063's `ArgReduceMeta` postmortem).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CrossMeta {
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<CrossMeta>() == 16);

const WGSL_CROSS_META: &str = r"struct CrossMeta {
    offsets: vec4<u32>,
}
";

struct CrossProductKernel;

fn cross_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> cmeta: CrossMeta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read> b: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= cmeta.offsets.x) {{
        return;
    }}
    let base = i * 3u;
    let ax = a[base];
    let ay = a[base + 1u];
    let az = a[base + 2u];
    let bx = b[base];
    let by = b[base + 1u];
    let bz = b[base + 2u];
    out[base] = ay * bz - az * by;
    out[base + 1u] = az * bx - ax * bz;
    out[base + 2u] = ax * by - ay * bx;
}}
"#,
        meta = WGSL_CROSS_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

/// WGPU device marker implementing [`CrossProductOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCrossProductOps;

impl<T> CrossProductOps<WgpuDevice, T> for WgpuCrossProductOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn cross_into(
        &self,
        device: &WgpuDevice,
        a: &WgpuBuffer<T>,
        b: &WgpuBuffer<T>,
        out: &WgpuBuffer<T>,
    ) -> Result<()> {
        let triples = validate_cross_product_lengths(a.len(), b.len(), out.len())?;
        if a.aliases(out) || b.aliases(out) {
            return Err(HephaestusError::DispatchFailed {
                message: "cross product output must not alias either input".to_string(),
            });
        }
        if triples == 0 {
            return Ok(());
        }
        let block_width = BlockWidth::DEFAULT;

        let meta = CrossMeta {
            offsets: [to_u32(triples, "triple count")?, 0, 0, 0],
        };
        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<CrossMeta>(1)?)?;
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let pipeline = cached_pipeline(
            device,
            (
                TypeId::of::<CrossProductKernel>(),
                TypeId::of::<T>(),
                block_width.get(),
            ),
            "hephaestus-cross-product",
            || cross_shader::<T>(block_width),
        );

        let groups = workgroups(triples, block_width)?;
        let mut entries = BindGroupEntries::with_capacity(4);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: meta_buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: a.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: b.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: out.buffer.as_entire_binding(),
        });
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-cross-product"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-cross-product"),
            });
        encode_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            groups,
            "hephaestus-cross-product",
        );
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
