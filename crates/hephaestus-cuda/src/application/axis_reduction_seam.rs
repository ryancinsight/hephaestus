//! CUDA implementation of the device-neutral axis-reduction seam.
//!
//! The kernels live in [`crate::application::reduction`] and the reusable
//! plans in [`crate::application::prepared_axis_reduction`]; this module only
//! adapts them to [`hephaestus_core::AxisReductionOps`] so a consumer — or the
//! conformance suite — can reduce along an axis without naming `CudaDevice`.
//! The prepared form borrows its operand pair, so re-dispatch observes writes
//! made to the bound input between dispatches (the seam's rebind contract).

use eunomia::Pod;
use hephaestus_core::{
    AxisReductionOps, BlockWidth, CombineExpr, CudaC, DialectScalar, IdentityToken, OpIdentity,
    ProdOp, Result, StridedView, SumOp,
};

use crate::application::prepared_axis_reduction::{
    PreparedAxisReduction, prepare_reduce_axis_into,
};
use crate::application::reduction::{mean_axis_into, prod_axis_into, reduce_axis_into};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Axis reductions for one CUDA device.
///
/// Zero-sized: kernels are cached on the device, so the seam holds no
/// prepared resources of its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaAxisReductionOps;

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a, T, const N: usize>(
    view: StridedView<'a, CudaBuffer<T>, N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}

impl<T> AxisReductionOps<CudaDevice, T> for CudaAxisReductionOps
where
    T: DialectScalar<CudaC> + Pod,
{
    type Dialect = CudaC;
    type Prepared<'op>
        = PreparedAxisReduction<'op, T>
    where
        T: 'op;

    #[inline]
    fn reduce_axis_into<Op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, CudaBuffer<T>, 2>,
    ) -> Result<()>
    where
        Op: CombineExpr<CudaC>,
        T: OpIdentity<Op> + IdentityToken<Op, CudaC>,
    {
        reduce_axis_into::<Op, T>(
            device,
            operand(input),
            axis,
            operand(output),
            BlockWidth::DEFAULT,
        )
    }

    #[inline]
    fn prod_axis_into(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, CudaBuffer<T>, 2>,
    ) -> Result<()>
    where
        T: OpIdentity<ProdOp> + IdentityToken<ProdOp, CudaC>,
    {
        prod_axis_into::<T>(
            device,
            operand(input),
            axis,
            operand(output),
            BlockWidth::DEFAULT,
        )
    }

    #[inline]
    fn mean_axis_into(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, CudaBuffer<T>, 2>,
    ) -> Result<()>
    where
        T: OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
    {
        mean_axis_into::<T>(
            device,
            operand(input),
            axis,
            operand(output),
            BlockWidth::DEFAULT,
        )
    }

    fn prepare_reduce_axis_into<'op, Op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'op, CudaBuffer<T>, 2>,
    ) -> Result<Self::Prepared<'op>>
    where
        Op: CombineExpr<CudaC>,
        T: OpIdentity<Op> + IdentityToken<Op, CudaC>,
    {
        prepare_reduce_axis_into::<Op, T>(
            device,
            operand(input),
            axis,
            operand(output),
            BlockWidth::DEFAULT,
        )
    }

    #[inline]
    fn dispatch_prepared(&self, device: &CudaDevice, prepared: &Self::Prepared<'_>) -> Result<()> {
        prepared.dispatch(device)
    }
}
