use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, DialectScalar, ElementwiseOps, HipC, Result, StridedView,
    TypedBinaryExpr, UnaryExpr,
};

use crate::application::prepared_strided_elementwise::{
    PreparedStridedBinary, PreparedStridedScalar, PreparedStridedUnary,
    prepare_binary_elementwise_strided_into, prepare_binary_elementwise_strided_typed_into,
    prepare_scalar_elementwise_strided_into, prepare_unary_elementwise_strided_into,
};
use crate::application::strided::StridedOperand;
use crate::{RocmBuffer, RocmDevice};

/// Provider-owned implementation of [`ElementwiseOps`] for ROCm/HIP.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmElementwiseOps;

impl<T> ElementwiseOps<RocmDevice, T> for RocmElementwiseOps
where
    T: DialectScalar<HipC> + Pod,
{
    type Dialect = HipC;
    type PreparedUnary<'op, const N: usize>
        = PreparedStridedUnary<'op, T>
    where
        T: 'op;
    type PreparedBinary<'op, const N: usize>
        = PreparedStridedBinary<'op, T>
    where
        T: 'op;
    type PreparedScalar<'op, const N: usize>
        = PreparedStridedScalar<'op, T>
    where
        T: 'op;
    type PreparedTypedBinary<'op, const N: usize>
        = PreparedStridedBinary<'op, T>
    where
        T: 'op;

    fn prepare_unary_into<'op, Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<T>, N>,
        output: StridedView<'op, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided_elementwise::MAX_STRIDED_RANK);
        }
        prepare_unary_elementwise_strided_into::<Op, T, N>(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
            StridedOperand {
                buffer: output.buffer,
                layout: output.layout,
            },
            BlockWidth::DEFAULT,
        )
    }

    fn dispatch_unary<const N: usize>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'op, RocmBuffer<T>, N>,
        rhs: StridedView<'op, RocmBuffer<T>, N>,
        output: StridedView<'op, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided_elementwise::MAX_STRIDED_RANK);
        }
        prepare_binary_elementwise_strided_into::<Op, T, N>(
            device,
            StridedOperand {
                buffer: lhs.buffer,
                layout: lhs.layout,
            },
            StridedOperand {
                buffer: rhs.buffer,
                layout: rhs.layout,
            },
            StridedOperand {
                buffer: output.buffer,
                layout: output.layout,
            },
            BlockWidth::DEFAULT,
        )
    }

    fn dispatch_binary<const N: usize>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<T>, N>,
        scalar: T,
        output: StridedView<'op, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided_elementwise::MAX_STRIDED_RANK);
        }
        prepare_scalar_elementwise_strided_into::<Op, T, N>(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input.layout,
            },
            scalar,
            StridedOperand {
                buffer: output.buffer,
                layout: output.layout,
            },
            BlockWidth::DEFAULT,
        )
    }

    fn dispatch_scalar<const N: usize>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedScalar<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'op, RocmBuffer<T>, N>,
        rhs: StridedView<'op, RocmBuffer<T>, N>,
        output: StridedView<'op, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
    {
        const {
            assert!(N <= crate::application::strided_elementwise::MAX_STRIDED_RANK);
        }
        prepare_binary_elementwise_strided_typed_into::<Op, T, N>(
            device,
            StridedOperand {
                buffer: lhs.buffer,
                layout: lhs.layout,
            },
            StridedOperand {
                buffer: rhs.buffer,
                layout: rhs.layout,
            },
            StridedOperand {
                buffer: output.buffer,
                layout: output.layout,
            },
            BlockWidth::DEFAULT,
        )
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}
