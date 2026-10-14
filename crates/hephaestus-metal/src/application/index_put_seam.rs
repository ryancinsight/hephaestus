//! Provider-owned n-D scatter-assignment seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, IndexPutOps, Result, StridedView, Wgsl};
use hephaestus_wgpu::WgpuIndexPutOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`IndexPutOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalIndexPutOps {
    inner: WgpuIndexPutOps,
}

impl<T> IndexPutOps<MetalDevice, T> for MetalIndexPutOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    #[allow(clippy::too_many_arguments)]
    fn index_put_into<const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, MetalBuffer<u32>, 1>,
        values: StridedView<'_, MetalBuffer<T>, N>,
        accumulate: bool,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.index_put_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            dim,
            StridedView::new(&index.buffer.inner, index.layout),
            StridedView::new(&values.buffer.inner, values.layout),
            accumulate,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
