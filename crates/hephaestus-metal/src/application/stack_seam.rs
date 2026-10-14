//! Provider-owned n-D stacking seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, StackOps, StridedView, Wgsl};
use hephaestus_wgpu::WgpuStackOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`StackOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalStackOps {
    inner: WgpuStackOps,
}

impl<T> StackOps<MetalDevice, T> for MetalStackOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    fn stack_into<const N: usize, const M: usize>(
        &self,
        device: &MetalDevice,
        inputs: &[StridedView<'_, MetalBuffer<T>, N>],
        axis: usize,
        output: StridedView<'_, MetalBuffer<T>, M>,
    ) -> Result<()>
    where
        leto::RankMarker<N>: leto::InsertAxis<N, LargerShape = [usize; M]>,
    {
        let views: Vec<StridedView<'_, _, N>> = inputs
            .iter()
            .map(|view| StridedView::new(&view.buffer.inner, view.layout))
            .collect();
        self.inner.stack_into(
            device.wgpu_device(),
            &views,
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
