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
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, EmbeddingOps, HephaestusError,
    IdentityToken, Result, SumOp, Wgsl, validate_embedding_gather_shape,
    validate_embedding_scatter_shape,
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

struct EmbeddingScatterKernel;

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

/// One thread per output row: row `r` sums `grad_out[i]` over the input
/// rows with `indices[i] == r`, in ascending `i` — the seam's documented
/// accumulation order. WGSL has no floating-point atomics, so an
/// input-parallel scatter would race on repeated indices; the scan keeps
/// every backend on the same order instead.
///
/// Out-of-range indices trip the same atomic error flag the gather kernel
/// uses; the offending contribution is skipped and the call fails after the
/// download, leaving `grad_table` partially accumulated — irrelevant once
/// the call returns an error.
fn scatter_shader<T: DialectScalar<Wgsl> + IdentityToken<SumOp, Wgsl>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> gmeta: GatherMeta;
@group(0) @binding(1) var<storage, read> grad_out: array<{ty}>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_table: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> error_flag: atomic<u32>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let r = gid.x;
    if (r >= gmeta.offsets.y) {{
        return;
    }}
    let dim = gmeta.offsets.x;
    let n = gmeta.offsets.z;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {{
        var acc: {ty} = {identity};
        for (var i: u32 = 0u; i < n; i = i + 1u) {{
            let idx = indices[i];
            if (idx >= gmeta.offsets.y) {{
                atomicStore(&error_flag, 1u);
                continue;
            }}
            if (idx == r) {{
                acc = acc + grad_out[i * dim + d];
            }}
        }}
        grad_table[r * dim + d] = grad_table[r * dim + d] + acc;
    }}
}}
"#,
        meta = WGSL_GATHER_META,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<SumOp, Wgsl>>::TOKEN,
    )
}

/// WGPU device marker implementing [`EmbeddingOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuEmbeddingOps;

/// Shared gather/scatter dispatch plumbing: uniform upload, pipeline fetch,
/// bind group, encode, submit, and atomic-flag download.
///
/// Both kernels take the same five bindings in the same order — uniform,
/// read-only source, indices, read-write destination, error flag — and the
/// same `[dim, num_embeddings, n]` triple; only the thread domain, the
/// pipeline, and the buffers bound to source/destination differ.
struct EmbeddingDispatch<'a, T> {
    device: &'a WgpuDevice,
    source: &'a WgpuBuffer<T>,
    indices: &'a WgpuBuffer<u32>,
    dest: &'a WgpuBuffer<T>,
    dim: u32,
    num_embeddings: u32,
    rows: u32,
    threads: usize,
    kernel: TypeId,
    entry: &'static str,
    make_source: fn(BlockWidth) -> String,
    error_verb: &'static str,
}

impl<T: Pod> EmbeddingDispatch<'_, T> {
    fn run(self) -> Result<()> {
        let block_width = BlockWidth::DEFAULT;
        let meta = GatherMeta {
            offsets: [self.dim, self.num_embeddings, self.rows, 0],
        };
        let meta_buffer = self
            .device
            .get_uniform_buffer(WgpuDevice::byte_size::<GatherMeta>(1)?)?;
        self.device
            .queue()
            .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

        let error_flag = self.device.alloc_zeroed::<u32>(1)?;

        let pipeline = cached_pipeline(
            self.device,
            (self.kernel, TypeId::of::<T>(), block_width.get()),
            self.entry,
            || (self.make_source)(block_width),
        );

        let groups = workgroups(self.threads, block_width)?;
        let mut entries = BindGroupEntries::with_capacity(5);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: meta_buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: self.source.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: self.indices.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: self.dest.buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: error_flag.buffer.as_entire_binding(),
        });
        let bind_group = self
            .device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.entry),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        let mut encoder =
            self.device
                .inner()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(self.entry),
                });
        encode_compute_pass(&mut encoder, &pipeline, &bind_group, groups, self.entry);
        self.device.queue().submit(Some(encoder.finish()));

        let mut flag_host = [0u32; 1];
        self.device.download(&error_flag, &mut flag_host)?;
        if flag_host[0] != 0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "embedding {} encountered an index out of range for \
                     {} embeddings",
                    self.error_verb, self.num_embeddings,
                ),
            });
        }
        Ok(())
    }
}

impl<T> EmbeddingOps<WgpuDevice, T> for WgpuEmbeddingOps
where
    T: DialectScalar<Wgsl> + Pod + IdentityToken<SumOp, Wgsl>,
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
        EmbeddingDispatch {
            device,
            source: table,
            indices,
            dest: output,
            dim: to_u32(embedding_dim, "embedding dimension")?,
            num_embeddings: to_u32(num_embeddings, "embedding count")?,
            rows: to_u32(rows, "dispatch size")?,
            threads: rows,
            kernel: TypeId::of::<EmbeddingGatherKernel>(),
            entry: "hephaestus-embedding-gather",
            make_source: gather_shader::<T>,
            error_verb: "gather",
        }
        .run()
    }

    fn scatter_add_into(
        &self,
        device: &WgpuDevice,
        grad_table: &WgpuBuffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &WgpuBuffer<u32>,
        grad_out: &WgpuBuffer<T>,
    ) -> Result<()> {
        let rows = validate_embedding_scatter_shape(
            grad_table.len(),
            num_embeddings,
            embedding_dim,
            indices.len(),
            grad_out.len(),
        )?;
        if rows == 0 {
            return Ok(());
        }
        if num_embeddings == 0 {
            // No output row exists for any thread to scan from, so the
            // kernel could not flag the indices — but every index is
            // trivially out of range for an empty table.
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "embedding scatter-add encountered an index out of range for \
                     {num_embeddings} embeddings"
                ),
            });
        }
        EmbeddingDispatch {
            device,
            source: grad_out,
            indices,
            dest: grad_table,
            dim: to_u32(embedding_dim, "embedding dimension")?,
            num_embeddings: to_u32(num_embeddings, "embedding count")?,
            rows: to_u32(rows, "dispatch size")?,
            threads: num_embeddings,
            kernel: TypeId::of::<EmbeddingScatterKernel>(),
            entry: "hephaestus-embedding-scatter",
            make_source: scatter_shader::<T>,
            error_verb: "scatter-add",
        }
        .run()
    }
}
