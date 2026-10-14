//! Provider-owned n-D inner-product seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, DotOps, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuDotOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`DotOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalDotOps {
    inner: WgpuDotOps,
}

impl<T> DotOps<MetalDevice, T> for MetalDotOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn dot_into<const N: usize, const M: usize>(
        &self,
        device: &MetalDevice,
        a: StridedView<'_, MetalBuffer<T>, N>,
        b: StridedView<'_, MetalBuffer<T>, M>,
        out: StridedView<'_, MetalBuffer<T>, 1>,
    ) -> Result<()> {
        self.inner.dot_into(
            device.wgpu_device(),
            StridedView::new(&a.buffer.inner, a.layout),
            StridedView::new(&b.buffer.inner, b.layout),
            StridedView::new(&out.buffer.inner, out.layout),
        )
    }
}
