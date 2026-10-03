//! WGPU implementation of [`hephaestus_core::EmbeddingOps`].
//!
//! One thread per output row. `indices` is itself device-resident data, so
//! an out-of-range index cannot be caught before dispatch the way a
//! host-known shape mismatch can — the kernel detects it and reports it
//! through a one-`u32` atomic error flag, downloaded after the dispatch
//! completes, rather than clamping or wrapping the index (error-handling
//! restraint: a data-dependent fault is surfaced, never silently masked).
//! The gather itself is skipped for any lane that trips the flag; its output
//! row is left at whatever `output` held on entry, which is irrelevant once
//! the call returns an error.

use std::any::TypeId;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, EmbeddingOps, HephaestusError, Result,
    Wgsl, validate_embedding_gather_shape,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::application::strided::to_u32;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGSL `GatherMeta` uniform: `[embedding_dim, num_embeddings, rows, _pad]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherMeta {
    offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<GatherMeta>() == 16);

const WGSL_GATHER_META: &str = r"struct GatherMeta {
    offsets: vec4<u32>,
}
";

struct EmbeddingGatherKernel;

fn gather_shader<T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> gmeta: GatherMeta;
@group(0) @binding(1) var<storage, read> emb_table: array<{ty}>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= gmeta.offsets.z) {{
        return;
    }}
    let dim = gmeta.offsets.x;
    let num_embeddings = gmeta.offsets.y;
    let idx = indices[i];
    if (idx >= num_embeddings) {{
        atomicStore(&error_flag, 1u);
        return;
    }}
    let table_base = idx * dim;
    let out_base = i * dim;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {{
        out[out_base + d] = emb_table[table_base + d];
    }}
}}
"#,
        meta = WGSL_GATHER_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
    )
}

/// WGPU device marker implementing [`EmbeddingOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuEmbeddingOps;

impl<T> EmbeddingOps<WgpuDevice, T> for WgpuEmbeddingOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn gather_into(
        &self,
        device: &WgpuDevice,
        table: &WgpuBuffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &WgpuBuffer<u32>,
        output: &WgpuBuffer<T>,
    ) -> Result<()> {
        let rows = validate_embedding_gather_shape(
            table.len(),
            num_embeddings,
            embedding_dim,
            indices.len(),
            output.len(),
        )?;
        if rows == 0 {
            return Ok(());
        }
        let block_width = BlockWidth::DEFAULT;

        let meta = GatherMeta {
            offsets: [
                to_u32(embedding_dim, "embedding dimension")?,
                to_u32(num_embeddings, "embedding count")?,
                to_u32(rows, "dispatch size")?,
                0,
            ],
        };
        let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<GatherMeta>(1)?)?;
        device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let error_flag = device.alloc_zeroed::<u32>(1)?;

        let pipeline = cached_pipeline(
            device,
            (
                TypeId::of::<EmbeddingGatherKernel>(),
                TypeId::of::<T>(),
                block_width.get(),
            ),
            "hephaestus-embedding-gather",
            || gather_shader::<T>(block_width),
        );

        let groups = workgroups(rows, block_width)?;
        let mut entries = BindGroupEntries::with_capacity(5);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: meta_buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: table.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: indices.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: output.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: error_flag.buffer.as_entire_binding(),
        });
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-embedding-gather"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-embedding-gather"),
            });
        encode_compute_pass(
            &mut encoder,
            &pipeline,
            &bind_group,
            groups,
            "hephaestus-embedding-gather",
        );
        device.queue().submit(Some(encoder.finish()));

        let mut flag_host = [0u32; 1];
        device.download(&error_flag, &mut flag_host)?;
        if flag_host[0] != 0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "embedding gather encountered an index out of range for \
                     {num_embeddings} embeddings"
                ),
            });
        }
        Ok(())
    }
}
