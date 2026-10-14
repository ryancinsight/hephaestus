//! Provider-owned n-D stable-sort seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, SortOps, StridedView, Wgsl};
use hephaestus_wgpu::WgpuSortOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`SortOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalSortOps {
    inner: WgpuSortOps,
}

impl<T> SortOps<MetalDevice, T> for MetalSortOps
where
    T: DialectScalar<Wgsl> + Pod + PartialOrd,
{
    type Dialect = Wgsl;

    fn sort_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        dim: usize,
        descending: bool,
        values: StridedView<'_, MetalBuffer<T>, N>,
        indices: StridedView<'_, MetalBuffer<u32>, N>,
    ) -> Result<()> {
        self.inner.sort_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            dim,
            descending,
            StridedView::new(&values.buffer.inner, values.layout),
            StridedView::new(&indices.buffer.inner, indices.layout),
        )
    }
}
