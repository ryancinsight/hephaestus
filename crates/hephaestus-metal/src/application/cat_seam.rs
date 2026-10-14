//! Provider-owned n-D concatenation seam for Metal.

use eunomia::Pod;
use hephaestus_core::{CatOps, DialectScalar, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuCatOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`CatOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalCatOps {
    inner: WgpuCatOps,
}

impl<T> CatOps<MetalDevice, T> for MetalCatOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn cat_into<const N: usize>(
        &self,
        device: &MetalDevice,
        inputs: &[StridedView<'_, MetalBuffer<T>, N>],
        axis: usize,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        let views: Vec<StridedView<'_, _, N>> = inputs
            .iter()
            .map(|view| StridedView::new(&view.buffer.inner, view.layout))
            .collect();
        self.inner.cat_into(
            device.wgpu_device(),
            &views,
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
