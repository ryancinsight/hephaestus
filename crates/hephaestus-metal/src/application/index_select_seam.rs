//! Provider-owned n-D index-selection seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, IndexSelectOps, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuIndexSelectOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`IndexSelectOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalIndexSelectOps {
    inner: WgpuIndexSelectOps,
}

impl<T> IndexSelectOps<MetalDevice, T> for MetalIndexSelectOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn index_select_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, MetalBuffer<u32>, 1>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.index_select_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            dim,
            StridedView::new(&index.buffer.inner, index.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
