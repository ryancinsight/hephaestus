use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, DialectScalar, ElementwiseOps, HipC, Result, StridedView,
    TypedBinaryExpr, UnaryExpr,
};

use crate::application::strided::StridedOperand;
use crate::application::strided_elementwise::{
    binary_elementwise_strided_into, binary_elementwise_strided_typed_into,
    unary_elementwise_strided_into,
};
use crate::{RocmBuffer, RocmDevice};

/// Provider-owned implementation of [`ElementwiseOps`] for ROCm/HIP.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmElementwiseOps;

/// Prepared elementwise state; the operation runs in `prepare` under ROCm's
/// synchronous execution model so `dispatch` is a no-op.
#[derive(Clone, Copy, Debug)]
pub struct RocmPreparedElementwise;

impl<T> ElementwiseOps<RocmDevice, T> for RocmElementwiseOps
where
    T: DialectScalar<HipC> + Pod,
{
    type Dialect = HipC;
    type PreparedUnary<const N: usize> = RocmPreparedElementwise;
    type PreparedBinary<const N: usize> = RocmPreparedElementwise;
    type PreparedTypedBinary<const N: usize> = RocmPreparedElementwise;

    fn prepare_unary_into<Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided_elementwise::MAX_STRIDED_RANK);
        }
        unary_elementwise_strided_into::<Op, T, N>(
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
        )?;
        Ok(RocmPreparedElementwise)
    }

    fn dispatch_unary<const N: usize>(
        &self,
        _device: &RocmDevice,
        _prepared: &Self::PreparedUnary<N>,
    ) -> Result<()> {
        Ok(())
    }

    fn prepare_binary_into<Op, const N: usize>(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'_, RocmBuffer<T>, N>,
        rhs: StridedView<'_, RocmBuffer<T>, N>,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided_elementwise::MAX_STRIDED_RANK);
        }
        binary_elementwise_strided_into::<Op, T, N>(
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
        )?;
        Ok(RocmPreparedElementwise)
    }

    fn dispatch_binary<const N: usize>(
        &self,
        _device: &RocmDevice,
        _prepared: &Self::PreparedBinary<N>,
    ) -> Result<()> {
        Ok(())
    }

    fn prepare_typed_binary_into<Op, const N: usize>(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'_, RocmBuffer<T>, N>,
        rhs: StridedView<'_, RocmBuffer<T>, N>,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
    {
        const {
            assert!(N <= crate::application::strided_elementwise::MAX_STRIDED_RANK);
        }
        binary_elementwise_strided_typed_into::<Op, T, N>(
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
        )?;
        Ok(RocmPreparedElementwise)
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        _device: &RocmDevice,
        _prepared: &Self::PreparedTypedBinary<N>,
    ) -> Result<()> {
        Ok(())
    }
}
