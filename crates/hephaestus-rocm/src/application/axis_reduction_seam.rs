//! ROCm/HIP implementation of the device-neutral axis-reduction seam.
//!
//! The kernels live in [`crate::application::axis_reduction`] and the
//! reusable plans in [`crate::application::prepared_axis_reduction`]; this
//! module only adapts them to [`hephaestus_core::AxisReductionOps`] so a
//! consumer — or the conformance suite — can reduce along an axis without
//! naming `RocmDevice`. The prepared form borrows its operand pair, so
//! re-dispatch observes writes made to the bound input between dispatches
//! (the seam's rebind contract).

use bytemuck::Pod;
use hephaestus_core::{
    AxisReductionOps, BlockWidth, CombineExpr, DialectScalar, HipC, IdentityToken, OpIdentity,
    ProdOp, Result, StridedView, SumOp,
};

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::axis_reduction::{mean_axis_into, prod_axis_into, reduce_axis_into};
use crate::application::prepared_axis_reduction::{
    PreparedAxisReduction, prepare_reduce_axis_into,
};
use crate::application::strided::StridedOperand;

/// Axis reductions for one ROCm device.
///
/// Zero-sized: kernels are cached on the device, so the seam holds no
/// prepared resources of its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmAxisReductionOps;

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a, T, const N: usize>(
    view: StridedView<'a, RocmBuffer<T>, N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}

impl<T> AxisReductionOps<RocmDevice, T> for RocmAxisReductionOps
where
    T: DialectScalar<HipC> + Pod,
{
    type Dialect = HipC;
    type Prepared<'op>
        = PreparedAxisReduction<'op, T>
    where
        T: 'op;

    #[inline]
    fn reduce_axis_into<Op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()>
    where
        Op: CombineExpr<HipC>,
        T: OpIdentity<Op> + IdentityToken<Op, HipC>,
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
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()>
    where
        T: OpIdentity<ProdOp> + IdentityToken<ProdOp, HipC>,
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
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()>
    where
        T: OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
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
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'op, RocmBuffer<T>, 2>,
    ) -> Result<Self::Prepared<'op>>
    where
        Op: CombineExpr<HipC>,
        T: OpIdentity<Op> + IdentityToken<Op, HipC>,
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
    fn dispatch_prepared(&self, device: &RocmDevice, prepared: &Self::Prepared<'_>) -> Result<()> {
        prepared.dispatch(device)
    }
}
