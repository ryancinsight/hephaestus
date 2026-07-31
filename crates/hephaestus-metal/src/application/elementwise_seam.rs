//! Provider-owned elementwise operation seam for Metal.

use hephaestus_core::{
    BinaryExpr, DialectScalar, ElementwiseOps, Result, StridedView, TypedBinaryExpr, UnaryExpr,
    Wgsl,
};
use hephaestus_wgpu::{PreparedElementwise, WgpuDevice, WgpuElementwiseOps};

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of [`ElementwiseOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalElementwiseOps {
    inner: WgpuElementwiseOps,
}

impl<T> ElementwiseOps<MetalDevice, T> for MetalElementwiseOps
where
    T: DialectScalar<Wgsl> + bytemuck::Pod + Send + Sync,
{
    type Dialect = Wgsl;
    type PreparedUnary<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;
    type PreparedBinary<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;
    type PreparedTypedBinary<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;

    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'op, MetalBuffer<T>, N>,
        output: StridedView<'op, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        self.inner.prepare_unary_into::<Op, N>(
            device.wgpu_device(),
            StridedView::new(&input.buffer.inner, input.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn dispatch_unary<const N: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()> {
        <WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::dispatch_unary::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        device: &MetalDevice,
        lhs: StridedView<'op, MetalBuffer<T>, N>,
        rhs: StridedView<'op, MetalBuffer<T>, N>,
        output: StridedView<'op, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        self.inner.prepare_binary_into::<Op, N>(
            device.wgpu_device(),
            StridedView::new(&lhs.buffer.inner, lhs.layout),
            StridedView::new(&rhs.buffer.inner, rhs.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn dispatch_binary<const N: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()> {
        <WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::dispatch_binary::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        device: &MetalDevice,
        lhs: StridedView<'op, MetalBuffer<T>, N>,
        rhs: StridedView<'op, MetalBuffer<T>, N>,
        output: StridedView<'op, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
    {
        self.inner.prepare_typed_binary_into::<Op, N>(
            device.wgpu_device(),
            StridedView::new(&lhs.buffer.inner, lhs.layout),
            StridedView::new(&rhs.buffer.inner, rhs.layout),
            StridedView::new(&output.buffer.inner, output.layout),
        )
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()> {
        <WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::dispatch_typed_binary::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }
}
