//! Provider-owned n-D gather seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, GatherOps, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuGatherOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`GatherOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalGatherOps {
    inner: WgpuGatherOps,
}

impl<T> GatherOps<MetalDevice, T> for MetalGatherOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn gather_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, MetalBuffer<u32>, N>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.gather_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            dim,
            StridedView::new(&index.buffer.inner, index.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
