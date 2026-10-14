//! Provider-owned n-D scatter-add seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, ScatterOps, StridedView, Wgsl};
use hephaestus_wgpu::WgpuScatterOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`ScatterOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalScatterOps {
    inner: WgpuScatterOps,
}

impl<T> ScatterOps<MetalDevice, T> for MetalScatterOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn scatter_add_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, MetalBuffer<u32>, N>,
        src: StridedView<'_, MetalBuffer<T>, N>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.scatter_add_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            dim,
            StridedView::new(&index.buffer.inner, index.layout),
            StridedView::new(&src.buffer.inner, src.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
