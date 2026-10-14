//! Provider-owned adaptive average/maximum pooling seam for Metal.

use eunomia::Pod;
use hephaestus_core::{
    AdaptivePoolingMode, AdaptivePoolingOps, DialectScalar, Result, StridedView, Wgsl,
};
use hephaestus_wgpu::WgpuAdaptivePoolingOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`AdaptivePoolingOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalAdaptivePoolingOps {
    inner: WgpuAdaptivePoolingOps,
}

impl<T> AdaptivePoolingOps<MetalDevice, T> for MetalAdaptivePoolingOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn adaptive_pool_axis_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        mode: AdaptivePoolingMode,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()> {
        self.inner.adaptive_pool_axis_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            mode,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
