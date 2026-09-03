//! Provider-owned axis-reduction seam for Metal.
//!
//! Metal delegates wholly to the WGPU implementation, matching
//! [`crate::MetalScanOps`]: the buffer wrapper is unwrapped to its inner WGPU
//! buffer and every method forwards to
//! [`hephaestus_wgpu::WgpuAxisReductionOps`] on the device's WGPU handle.

use hephaestus_core::{
    AxisReductionOps, CombineExpr, DialectScalar, IdentityToken, OpIdentity, ProdOp, Result,
    StridedView, SumOp, Wgsl,
};
use hephaestus_wgpu::{PreparedAxisReduction, WgpuAxisReductionOps, WgpuDevice};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`AxisReductionOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalAxisReductionOps {
    inner: WgpuAxisReductionOps,
}

impl<T> AxisReductionOps<MetalDevice, T> for MetalAxisReductionOps
where
    T: DialectScalar<Wgsl> + eunomia::Pod,
{
    type Dialect = Wgsl;
    type Prepared<'op>
        = PreparedAxisReduction<T>
    where
        T: 'op;

    fn reduce_axis_into<Op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()>
    where
        Op: CombineExpr<Wgsl>,
        T: OpIdentity<Op> + IdentityToken<Op, Wgsl>,
    {
        self.inner.reduce_axis_into::<Op>(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn prod_axis_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()>
    where
        T: OpIdentity<ProdOp> + IdentityToken<ProdOp, Wgsl>,
    {
        self.inner.prod_axis_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn mean_axis_into(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, MetalBuffer<T>, 2>,
    ) -> Result<()>
    where
        T: OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
    {
        self.inner.mean_axis_into(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn prepare_reduce_axis_into<'op, Op>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'op, MetalBuffer<T>, 2>,
    ) -> Result<Self::Prepared<'op>>
    where
        Op: CombineExpr<Wgsl>,
        T: OpIdentity<Op> + IdentityToken<Op, Wgsl>,
    {
        self.inner.prepare_reduce_axis_into::<Op>(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            axis,
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn dispatch_prepared(&self, device: &MetalDevice, prepared: &Self::Prepared<'_>) -> Result<()> {
        <WgpuAxisReductionOps as AxisReductionOps<WgpuDevice, T>>::dispatch_prepared(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }
}
