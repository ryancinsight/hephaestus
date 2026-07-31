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
    type PreparedUnary<const N: usize> = PreparedElementwise;
    type PreparedBinary<const N: usize> = PreparedElementwise;
    type PreparedTypedBinary<const N: usize> = PreparedElementwise;

    fn prepare_unary_into<Op, const N: usize>(
        &self,
        device: &MetalDevice,
        input: StridedView<'_, MetalBuffer<T>, N>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<N>>
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
        prepared: &Self::PreparedUnary<N>,
    ) -> Result<()> {
        <WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::dispatch_unary::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }

    fn prepare_binary_into<Op, const N: usize>(
        &self,
        device: &MetalDevice,
        lhs: StridedView<'_, MetalBuffer<T>, N>,
        rhs: StridedView<'_, MetalBuffer<T>, N>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<N>>
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
        prepared: &Self::PreparedBinary<N>,
    ) -> Result<()> {
        <WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::dispatch_binary::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }

    fn prepare_typed_binary_into<Op, const N: usize>(
        &self,
        device: &MetalDevice,
        lhs: StridedView<'_, MetalBuffer<T>, N>,
        rhs: StridedView<'_, MetalBuffer<T>, N>,
        output: StridedView<'_, MetalBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<N>>
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
        prepared: &Self::PreparedTypedBinary<N>,
    ) -> Result<()> {
        <WgpuElementwiseOps as ElementwiseOps<WgpuDevice, T>>::dispatch_typed_binary::<N>(
            &self.inner,
            device.wgpu_device(),
            prepared,
        )
    }
}
