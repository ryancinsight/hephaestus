use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, DialectScalar, ElementwiseOps, Result, StridedView, TypedBinaryExpr, UnaryExpr,
    Wgsl,
};

use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use super::binary::prepare_binary_inner;
use super::dispatch::dispatch_prepared;
use super::scalar::prepare_scalar_inner;
use super::typed::SeamTypedExpr;
use super::types::PreparedElementwise;
use super::types::WgpuElementwiseOps;
use super::unary::prepare_unary_inner;

impl<T> ElementwiseOps<WgpuDevice, T> for WgpuElementwiseOps
where
    T: DialectScalar<Wgsl> + Pod + Send + Sync + 'static,
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
    type PreparedScalar<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;
    type PreparedTypedBinary<'op, const N: usize>
        = PreparedElementwise
    where
        T: 'op;

    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<T>, N>,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        prepare_unary_inner::<Op, T, N>(device, input, output)
    }

    fn dispatch_unary<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-unary")
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'op, WgpuBuffer<T>, N>,
        rhs: StridedView<'op, WgpuBuffer<T>, N>,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_binary_inner::<Op, T, N>(device, lhs, rhs, output, <Op as BinaryExpr<Wgsl>>::EXPR)
    }

    fn dispatch_binary<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-binary")
    }

    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<T>, N>,
        scalar: T,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_scalar_inner::<Op, T, N>(device, input, scalar, output)
    }

    fn dispatch_scalar<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedScalar<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-scalar")
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'op, WgpuBuffer<T>, N>,
        rhs: StridedView<'op, WgpuBuffer<T>, N>,
        output: StridedView<'op, WgpuBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
        T: DialectScalar<Self::Dialect>,
    {
        prepare_binary_inner::<SeamTypedExpr<Op, T>, T, N>(
            device,
            lhs,
            rhs,
            output,
            <Op as TypedBinaryExpr<Wgsl, T>>::EXPR,
        )
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_prepared::<N>(device, prepared, "hephaestus-seam-typed-binary")
    }
}
