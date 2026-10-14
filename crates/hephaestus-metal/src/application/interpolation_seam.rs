//! Provider-owned rank-2 axis-resampling seam for Metal.

use eunomia::Pod;
use hephaestus_core::{
    DialectScalar, InterpolationMode, InterpolationOps, Result, StridedView, Wgsl,
};
use hephaestus_wgpu::WgpuInterpolationOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`InterpolationOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalInterpolationOps {
    inner: WgpuInterpolationOps,
}

impl<T> InterpolationOps<MetalDevice, T> for MetalInterpolationOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn interpolate_axis_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        mode: InterpolationMode,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()> {
        self.inner.interpolate_axis_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            mode,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
