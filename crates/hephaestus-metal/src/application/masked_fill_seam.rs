//! Provider-owned n-D masked-fill seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, MaskedFillOps, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuMaskedFillOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`MaskedFillOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalMaskedFillOps {
    inner: WgpuMaskedFillOps,
}

impl<T> MaskedFillOps<MetalDevice, T> for MetalMaskedFillOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;

    fn masked_fill_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        mask: StridedView<'_, MetalBuffer<T>, N>,
        value: T,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.masked_fill_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            StridedView::new(&mask.buffer.inner, mask.layout),
            value,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
