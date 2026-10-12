//! ROCm implementation of [`hephaestus_core::EmbeddingOps`].
//!
//! One thread per output row. `indices` is itself device-resident data, so
//! an out-of-range index cannot be caught before dispatch the way a
//! host-known shape mismatch can — the kernel detects it and reports it
//! through a one-`u32` flag written with `atomicExch`, downloaded after the
//! dispatch completes, rather than clamping or wrapping the index
//! (error-handling restraint: a data-dependent fault is surfaced, never
//! silently masked). The faulting lane's own row is skipped; its output is
//! left at whatever the destination held on entry, which is irrelevant once
//! the call returns an error.
//!
//! The backward pass scans rather than scattering atomically: each output
//! row sums its input rows in ascending input order, the seam's documented
//! accumulation order, so every backend agrees bitwise.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, EmbeddingOps, HephaestusError, HipC,
    IdentityToken, Result, SumOp, validate_embedding_gather_shape,
    validate_embedding_scatter_shape,
};

use crate::RocmDevice;
use crate::application::linalg::to_u32;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

fn gather_source<T: DialectScalar<HipC>>() -> String {
    format!(
        r#"{prelude}extern "C" __global__ void embedding_gather_kernel(
    const {ty}* table,
    const unsigned int* indices,
    {ty}* out,
    unsigned int* error_flag,
    unsigned int embedding_dim,
    unsigned int num_embeddings,
    unsigned int rows
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {{
        unsigned int idx = indices[i];
        if (idx >= num_embeddings) {{
            atomicExch(error_flag, 1u);
            return;
        }}
        unsigned int table_base = idx * embedding_dim;
        unsigned int out_base = i * embedding_dim;
        for (unsigned int d = 0u; d < embedding_dim; ++d) {{
            out[out_base + d] = table[table_base + d];
        }}
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
    )
}

fn scatter_source<T: DialectScalar<HipC> + IdentityToken<SumOp, HipC>>() -> String {
    format!(
        r#"{prelude}extern "C" __global__ void embedding_scatter_kernel(
    const {ty}* grad_out,
    const unsigned int* indices,
    {ty}* grad_table,
    unsigned int* error_flag,
    unsigned int embedding_dim,
    unsigned int num_embeddings,
    unsigned int rows
) {{
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_embeddings) {{
        for (unsigned int d = 0u; d < embedding_dim; ++d) {{
            {ty} acc = {identity};
            for (unsigned int i = 0u; i < rows; ++i) {{
                unsigned int idx = indices[i];
                if (idx >= num_embeddings) {{
                    atomicExch(error_flag, 1u);
                    continue;
                }}
                if (idx == r) {{
                    acc = acc + grad_out[i * embedding_dim + d];
                }}
            }}
            grad_table[r * embedding_dim + d] = grad_table[r * embedding_dim + d] + acc;
        }}
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        identity = <T as IdentityToken<SumOp, HipC>>::TOKEN,
    )
}

/// Shared gather/scatter dispatch plumbing: kernel fetch, argument
/// marshalling, launch, and error-flag download.
///
/// Both kernels take the same seven arguments in the same order; only the
/// thread domain, the pipeline, and the buffers bound to source/destination
/// differ.
struct EmbeddingLaunch<'a, F> {
    device: &'a RocmDevice,
    source: DevicePtr,
    indices: DevicePtr,
    dest: DevicePtr,
    dim: u32,
    num: u32,
    rows: u32,
    threads: usize,
    key: PipelineKey,
    entry: &'static str,
    make_source: F,
    error_verb: &'static str,
    num_embeddings: usize,
}

impl<F: FnOnce() -> String> EmbeddingLaunch<'_, F> {
    fn run(self) -> Result<()> {
        let width = BlockWidth::DEFAULT;
        let kernel = cached_kernel(self.device, self.key, self.entry, self.make_source)?;

        let mut source_ptr: DevicePtr = self.source;
        let mut indices_ptr: DevicePtr = self.indices;
        let mut dest_ptr: DevicePtr = self.dest;
        let error_flag = self.device.alloc_zeroed::<u32>(1)?;
        let mut flag_ptr: DevicePtr = error_flag.raw();
        let mut dim_val = self.dim;
        let mut num_val = self.num;
        let mut rows_val = self.rows;

        // Argument list mirrors the embedding kernels' (const T*,
        // const unsigned int*, T*, unsigned int*, unsigned int x3).
        let mut args: [*mut core::ffi::c_void; 7] = [
            (&mut source_ptr as *mut DevicePtr).cast(),
            (&mut indices_ptr as *mut DevicePtr).cast(),
            (&mut dest_ptr as *mut DevicePtr).cast(),
            (&mut flag_ptr as *mut DevicePtr).cast(),
            (&mut dim_val as *mut u32).cast(),
            (&mut num_val as *mut u32).cast(),
            (&mut rows_val as *mut u32).cast(),
        ];
        launch_kernel(
            self.device,
            &kernel,
            LaunchConfig::linear(grid_size(self.threads, width)?, width),
            &mut args,
        )?;

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

/// ROCm device marker implementing [`EmbeddingOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmEmbeddingOps;

impl<T> EmbeddingOps<RocmDevice, T> for RocmEmbeddingOps
where
    T: DialectScalar<HipC> + Pod + IdentityToken<SumOp, HipC>,
{
    fn gather_into(
        &self,
        device: &RocmDevice,
        table: &RocmBuffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &RocmBuffer<u32>,
        output: &RocmBuffer<T>,
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
        EmbeddingLaunch {
            device,
            source: table.raw(),
            indices: indices.raw(),
            dest: output.raw(),
            dim: to_u32(embedding_dim, "embedding dimension")?,
            num: to_u32(num_embeddings, "embedding count")?,
            rows: to_u32(rows, "embedding row count")?,
            threads: rows,
            key: PipelineKey::EmbeddingGather {
                scalar: TypeId::of::<T>(),
                width: BlockWidth::DEFAULT.get(),
            },
            entry: "embedding_gather_kernel",
            make_source: gather_source::<T>,
            error_verb: "gather",
            num_embeddings,
        }
        .run()
    }

    fn scatter_add_into(
        &self,
        device: &RocmDevice,
        grad_table: &RocmBuffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &RocmBuffer<u32>,
        grad_out: &RocmBuffer<T>,
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
        EmbeddingLaunch {
            device,
            source: grad_out.raw(),
            indices: indices.raw(),
            dest: grad_table.raw(),
            dim: to_u32(embedding_dim, "embedding dimension")?,
            num: to_u32(num_embeddings, "embedding count")?,
            rows: to_u32(rows, "embedding row count")?,
            threads: num_embeddings,
            key: PipelineKey::EmbeddingScatter {
                scalar: TypeId::of::<T>(),
                width: BlockWidth::DEFAULT.get(),
            },
            entry: "embedding_scatter_kernel",
            make_source: scatter_source::<T>,
            error_verb: "scatter-add",
            num_embeddings,
        }
        .run()
    }
}
