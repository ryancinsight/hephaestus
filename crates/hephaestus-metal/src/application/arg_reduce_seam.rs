//! Provider-owned rank-2 argmax/argmin seam for Metal.

use eunomia::Pod;
use hephaestus_core::{ArgReduceOps, DialectScalar, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuArgReduceOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`ArgReduceOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalArgReduceOps {
    inner: WgpuArgReduceOps,
}

impl<T> ArgReduceOps<MetalDevice, T> for MetalArgReduceOps
where
    T: DialectScalar<Wgsl> + Pod + PartialOrd,
{
    fn argmax_axis_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, MetalBuffer<u32>, 2>,
    ) -> Result<()> {
        self.inner.argmax_axis_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn argmin_axis_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, MetalBuffer<u32>, 2>,
    ) -> Result<()> {
        self.inner.argmin_axis_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
