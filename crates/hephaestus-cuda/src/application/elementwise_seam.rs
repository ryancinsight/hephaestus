use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, CudaC, DialectScalar, ElementwiseOps, Result, StridedView,
    TypedBinaryExpr, UnaryExpr,
};

use crate::application::prepared_strided_elementwise::{
    PreparedStridedBinary, PreparedStridedScalar, PreparedStridedUnary,
    prepare_binary_elementwise_strided_into, prepare_binary_elementwise_strided_typed_into,
    prepare_scalar_elementwise_strided_into, prepare_unary_elementwise_strided_into,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned implementation of [`ElementwiseOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaElementwiseOps;

impl<T> ElementwiseOps<CudaDevice, T> for CudaElementwiseOps
where
    T: DialectScalar<CudaC> + Pod,
{
    type Dialect = CudaC;
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
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<T>, N>,
        output: StridedView<'op, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedUnary<'op, N>>
    where
        Op: UnaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
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
        device: &CudaDevice,
        prepared: &Self::PreparedUnary<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_binary_into<'op, Op, const N: usize>(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'op, CudaBuffer<T>, N>,
        rhs: StridedView<'op, CudaBuffer<T>, N>,
        output: StridedView<'op, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedBinary<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
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
        device: &CudaDevice,
        prepared: &Self::PreparedBinary<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_scalar_into<'op, Op, const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<T>, N>,
        scalar: T,
        output: StridedView<'op, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedScalar<'op, N>>
    where
        Op: BinaryExpr<Self::Dialect>,
    {
        const {
            assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
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
        device: &CudaDevice,
        prepared: &Self::PreparedScalar<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_typed_binary_into<'op, Op, const N: usize>(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'op, CudaBuffer<T>, N>,
        rhs: StridedView<'op, CudaBuffer<T>, N>,
        output: StridedView<'op, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedTypedBinary<'op, N>>
    where
        Op: TypedBinaryExpr<Self::Dialect, T>,
    {
        const {
            assert!(N <= crate::application::strided::MAX_STRIDED_RANK);
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
        device: &CudaDevice,
        prepared: &Self::PreparedTypedBinary<'_, N>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}
