//! Provider-owned n-D nonzero-coordinate seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, NonzeroOps, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuNonzeroOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`NonzeroOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalNonzeroOps {
    inner: WgpuNonzeroOps,
}

impl<T> NonzeroOps<MetalDevice, T> for MetalNonzeroOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;

    fn nonzero_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        coords: StridedView<'_, MetalBuffer<u32>, 2>,
        count: StridedView<'_, MetalBuffer<u32>, 1>,
    ) -> Result<()> {
        self.inner.nonzero_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            StridedView::new(&coords.buffer.inner, coords.layout),
            StridedView::new(&count.buffer.inner, count.layout),
        )
    }
}
