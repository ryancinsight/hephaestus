//! Provider-owned embedding-table gather seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, EmbeddingOps, IdentityToken, Result, SumOp, Wgsl};
use hephaestus_wgpu::WgpuEmbeddingOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`EmbeddingOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalEmbeddingOps {
    inner: WgpuEmbeddingOps,
}

impl<T> EmbeddingOps<MetalDevice, T> for MetalEmbeddingOps
where
    T: DialectScalar<Wgsl> + Pod + IdentityToken<SumOp, Wgsl>,
{
    fn gather_into(
        &self,
        device: &MetalDevice,
        table: &MetalBuffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &MetalBuffer<u32>,
        output: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.gather_into(
            device.wgpu_device(),
            &table.inner,
            num_embeddings,
            embedding_dim,
            &indices.inner,
            &output.inner,
        )
    }

    fn scatter_add_into(
        &self,
        device: &MetalDevice,
        grad_table: &MetalBuffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &MetalBuffer<u32>,
        grad_out: &MetalBuffer<T>,
    ) -> Result<()> {
        self.inner.scatter_add_into(
            device.wgpu_device(),
            &grad_table.inner,
            num_embeddings,
            embedding_dim,
            &indices.inner,
            &grad_out.inner,
        )
    }
}
