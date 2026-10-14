//! Provider-owned rank-2 top-k selection seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, StridedView, TopKOps, Wgsl};
use hephaestus_wgpu::WgpuTopKOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`TopKOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalTopKOps {
    inner: WgpuTopKOps,
}

impl<T> TopKOps<MetalDevice, T> for MetalTopKOps
where
    T: DialectScalar<Wgsl> + Pod + PartialOrd,
{
    fn topk_axis_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        k: usize,
        values: StridedView<'_, MetalBuffer<T>, 2>,
        indices: StridedView<'_, MetalBuffer<u32>, 2>,
    ) -> Result<()> {
        self.inner.topk_axis_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            k,
            StridedView::new(&values.buffer.inner, values.layout),
            StridedView::new(&indices.buffer.inner, indices.layout),
        )
    }
}
