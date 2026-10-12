//! Provider-owned n-D conditional-selection seam for Metal.

use eunomia::Pod;
use hephaestus_core::{DialectScalar, Result, StridedView, Wgsl, WhereOps};
use hephaestus_wgpu::WgpuWhereOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`WhereOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalWhereOps {
    inner: WgpuWhereOps,
}

impl<T> WhereOps<MetalDevice, T> for MetalWhereOps
where
    T: DialectScalar<Wgsl> + Pod,
{
    type Dialect = Wgsl;

    fn where_into<const N: usize>(
        &self,
        device: &MetalDevice,
        cond: StridedView<'_, MetalBuffer<T>, N>,
        on_true: StridedView<'_, MetalBuffer<T>, N>,
        on_false: StridedView<'_, MetalBuffer<T>, N>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<()> {
        self.inner.where_into(
            device.wgpu_device(),
            StridedView::new(&cond.buffer.inner, cond.layout),
            StridedView::new(&on_true.buffer.inner, on_true.layout),
            StridedView::new(&on_false.buffer.inner, on_false.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
