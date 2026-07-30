use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, CudaC, DialectScalar, ElementwiseOps, Result, StridedView,
    TypedBinaryExpr, UnaryExpr,
};

use crate::application::strided::{
    StridedOperand, binary_elementwise_strided_into, binary_elementwise_strided_typed_into,
    unary_elementwise_strided_into,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned implementation of [`ElementwiseOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaElementwiseOps;

/// Prepared elementwise state; the operation runs in `prepare` under CUDA's
/// synchronous execution model so `dispatch` is a no-op.
#[derive(Clone, Copy, Debug)]
pub struct CudaPreparedElementwise;

impl<T> ElementwiseOps<CudaDevice, T> for CudaElementwiseOps
where
    T: DialectScalar<CudaC> + Pod,
{
    type Dialect = CudaC;
    type PreparedUnary<const N: usize> = CudaPreparedElementwise;
    type PreparedBinary<const N: usize> = CudaPreparedElementwise;
    type PreparedTypedBinary<const N: usize> = CudaPreparedElementwise;

    fn prepare_unary_into<Op, const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
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
        Ok(CudaPreparedElementwise)
    }

    fn dispatch_unary<const N: usize>(
        &self,
        _device: &CudaDevice,
        _prepared: &Self::PreparedUnary<N>,
    ) -> Result<()> {
        Ok(())
    }

    fn prepare_binary_into<Op, const N: usize>(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'_, CudaBuffer<T>, N>,
        rhs: StridedView<'_, CudaBuffer<T>, N>,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
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
        Ok(CudaPreparedElementwise)
    }

    fn dispatch_binary<const N: usize>(
        &self,
        _device: &CudaDevice,
        _prepared: &Self::PreparedBinary<N>,
    ) -> Result<()> {
        Ok(())
    }

    fn prepare_typed_binary_into<Op, const N: usize>(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'_, CudaBuffer<T>, N>,
        rhs: StridedView<'_, CudaBuffer<T>, N>,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
    {
        const {
            assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
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
        Ok(CudaPreparedElementwise)
    }

    fn dispatch_typed_binary<const N: usize>(
        &self,
        _device: &CudaDevice,
        _prepared: &Self::PreparedTypedBinary<N>,
    ) -> Result<()> {
        Ok(())
    }
}
