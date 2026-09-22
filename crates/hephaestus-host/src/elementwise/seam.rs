//! The [`ElementwiseOps`] implementation over the prepared forms.

use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, DialectScalar, ElementwiseOps, Host, Result, StridedView, TypedBinaryExpr,
    UnaryExpr,
};

use super::binary::{HostPreparedBinary, dispatch_binary, prepare_binary};
use super::dispatch::ElementwiseDispatch;
use super::scalar::{HostPreparedScalar, dispatch_scalar, prepare_scalar};
use super::unary::{HostPreparedUnary, dispatch_unary, prepare_unary};
use crate::{HostBuffer, HostDevice};

/// Unary and binary elementwise operations for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostElementwiseOps;

impl<T> ElementwiseOps<HostDevice, T> for HostElementwiseOps
where
    T: Pod + ElementwiseDispatch,
{
    type Dialect = Host;
    type PreparedUnary<'op, const N: usize>
        = HostPreparedUnary<'op, T, N>
    where
        T: 'op;
    type PreparedBinary<'op, const N: usize>
        = HostPreparedBinary<'op, T, N>
    where
        T: 'op;
    type PreparedScalar<'op, const N: usize>
        = HostPreparedScalar<'op, T, N>
    where
        T: 'op;
    type PreparedTypedBinary<'op, const N: usize>
        = HostPreparedBinary<'op, T, N>
    where
        T: 'op;

    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<T>, N>,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        prepare_unary::<Op, T, N>(input, output)
    }

    fn dispatch_unary<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()> {
        dispatch_unary::<T, N>(prepared)
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        lhs: StridedView<'op, HostBuffer<T>, N>,
        rhs: StridedView<'op, HostBuffer<T>, N>,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_binary(
            lhs,
            rhs,
            output,
            T::binary_fn::<Op>(),
            core::any::type_name::<Op>(),
        )
    }

    fn dispatch_binary<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_binary::<T, N>(prepared)
    }

    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'op, HostBuffer<T>, N>,
        scalar: T,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        prepare_scalar(
            input,
            scalar,
            output,
            T::binary_fn::<Op>(),
            core::any::type_name::<Op>(),
        )
    }

    fn dispatch_scalar<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedScalar<'_, N>,
    ) -> Result<()> {
        dispatch_scalar::<T, N>(prepared)
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        _device: &HostDevice,
        lhs: StridedView<'op, HostBuffer<T>, N>,
        rhs: StridedView<'op, HostBuffer<T>, N>,
        output: StridedView<'op, HostBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
        T: DialectScalar<Self::Dialect>,
    {
        prepare_binary(
            lhs,
            rhs,
            output,
            T::typed_binary_fn::<Op>(),
            core::any::type_name::<Op>(),
        )
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()> {
        dispatch_binary::<T, N>(prepared)
    }
}
