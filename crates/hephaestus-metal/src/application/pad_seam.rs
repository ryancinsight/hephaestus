//! Provider-owned n-D padding seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, PadOps, PadWidth, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuPadOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`PadOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalPadOps {
    inner: WgpuPadOps,
}

impl<T> PadOps<MetalDevice, T> for MetalPadOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn pad_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        width: PadWidth<N>,
        fill: T,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.pad_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            width,
            fill,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
