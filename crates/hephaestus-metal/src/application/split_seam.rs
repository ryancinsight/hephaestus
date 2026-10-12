//! Provider-owned n-D splitting seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, SplitOps, StridedView, Wgsl};
use hephaestus_wgpu::WgpuSplitOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`SplitOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalSplitOps {
    inner: WgpuSplitOps,
}

impl<T> SplitOps<MetalDevice, T> for MetalSplitOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn split_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        axis: usize,
        sizes: &[usize],
        outputs: &[StridedView<'_, MetalBuffer<T>, N>],
    ) -> Result<()> {
        let views: Vec<StridedView<'_, _, N>> = outputs
            .iter()
            .map(|view| StridedView::new(&view.buffer.inner, view.layout))
            .collect();
        self.inner.split_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            sizes,
            &views,
        )
    }
}
