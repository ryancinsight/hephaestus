//! WGPU implementation of the device-neutral axis-reduction seam.
//!
//! The kernels themselves live in [`crate::application::reduction`]; this module
//! only adapts them to [`AxisReductionOps`] so a consumer — or a conformance
//! suite — can reduce along an axis without naming `WgpuDevice`. It is the
//! reduction counterpart of [`crate::WgpuSparseOps`] and [`crate::WgpuVectorOps`].

use hephaestus_core::{
    AxisReductionOps, BlockWidth, CombineExpr, DialectScalar, IdentityToken, OpIdentity, ProdOp,
    Result, StridedView, Wgsl,
};

use crate::application::reduction::{PreparedAxisReduction, prepare_reduce_axis_into, prod_axis_into};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Axis reductions for one WGPU device.
///
/// Zero-sized: reduction pipelines are cached on the device, so the seam holds
/// no prepared resources of its own. It exists as a value rather than a set of
/// free functions so consumers hold one object per concern, matching
/// [`crate::WgpuSparseOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuAxisReductionOps;

/// Convert the device-neutral view into this backend's operand pair.
///
/// The two types are the same borrowed pair; only the buffer type is spelled
/// differently, so this is a field move the optimizer erases.
#[inline]
fn operand<'a, T>(view: StridedView<'a, WgpuBuffer<T>, 2>) -> StridedOperand<'a, T, 2> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}

impl<T> AxisReductionOps<WgpuDevice, T> for WgpuAxisReductionOps
where
    T: DialectScalar<Wgsl> + bytemuck::Pod,
{
    type Dialect = Wgsl;
    type Prepared = PreparedAxisReduction<T>;

    #[inline]
    fn prod_axis_into(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<()>
    where
        T: OpIdentity<ProdOp> + IdentityToken<ProdOp, Wgsl>,
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
    fn prepare_reduce_axis_into<Op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<Self::Prepared>
    where
        Op: CombineExpr<Wgsl>,
        T: OpIdentity<Op> + IdentityToken<Op, Wgsl>,
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
    fn dispatch_prepared(
        &self,
        device: &WgpuDevice,
        prepared: &Self::Prepared,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}
