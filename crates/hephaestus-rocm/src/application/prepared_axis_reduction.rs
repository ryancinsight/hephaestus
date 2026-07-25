//! Reusable rank-2 axis reduction plans for ROCm.

use std::rc::Rc;

use bytemuck::Pod;
use hephaestus_core::{
    AxisReductionDispatch, BlockWidth, CombineExpr, DialectScalar, HipC, IdentityToken, MaxOp,
    MinOp, OpIdentity, Result, SumOp,
};

use crate::RocmDevice;
use crate::application::axis_reduction::{
    axis_len, axis_reduction_shader_source, launch_with_meta, mean_axis_shader_source,
    plan_dispatch, reject_empty_axis,
};
use crate::application::pipeline::{PipelineKey, RocmKernel, cached_kernel};
use crate::application::strided::StridedOperand;

/// A reusable ROCm rank-2 axis reduction over fixed input and output views.
pub struct PreparedAxisReduction<'a, T> {
    input: StridedOperand<'a, T, 2>,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
    dispatch: Option<AxisReductionDispatch>,
    kernel: Option<Rc<RocmKernel>>,
}

impl<T> PreparedAxisReduction<'_, T> {
    /// Dispatch the prepared axis reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the native HIP launch fails.
    pub fn dispatch(&self, device: &RocmDevice) -> Result<()> {
        let (Some(dispatch), Some(kernel)) = (self.dispatch, self.kernel.as_ref()) else {
            return Ok(());
        };
        launch_with_meta(
            device,
            kernel,
            dispatch,
            self.input.buffer.raw(),
            self.output.buffer.raw(),
            self.width,
        )
    }
}

fn empty_prepared_axis_reduction<'a, T>(
    input: StridedOperand<'a, T, 2>,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> PreparedAxisReduction<'a, T> {
    PreparedAxisReduction {
        input,
        output,
        width,
        dispatch: None,
        kernel: None,
    }
}

fn prepared_axis_reduction<'a, T>(
    input: StridedOperand<'a, T, 2>,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
    dispatch: AxisReductionDispatch,
    kernel: Rc<RocmKernel>,
) -> PreparedAxisReduction<'a, T> {
    PreparedAxisReduction {
        input,
        output,
        width,
        dispatch: Some(dispatch),
        kernel: Some(kernel),
    }
}

/// Submit several prepared ROCm axis reductions in order.
///
/// # Errors
///
/// Returns the first native launch error encountered.
pub fn submit_prepared_axis_reduction_batch<T>(
    device: &RocmDevice,
    reductions: &[&PreparedAxisReduction<'_, T>],
) -> Result<()> {
    for reduction in reductions {
        reduction.dispatch(device)?;
    }
    Ok(())
}

/// Prepare a generic ROCm rank-2 axis reduction into fixed output storage.
///
/// # Errors
///
/// Returns a typed error when the shared axis planner rejects the axis, shape,
/// layout, width, aliasing, or buffer-size contract, or when kernel
/// compilation fails.
pub fn prepare_reduce_axis_into<'a, Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    let Some(dispatch) = plan_dispatch(input, axis, output, width)? else {
        return Ok(empty_prepared_axis_reduction(input, output, width));
    };
    let kernel = cached_kernel(
        device,
        PipelineKey::AxisReduction {
            op: core::any::TypeId::of::<Op>(),
            scalar: core::any::TypeId::of::<T>(),
            axis,
            width: width.get(),
        },
        "axis_reduction_kernel",
        axis_reduction_shader_source::<Op, T>,
    )?;
    Ok(prepared_axis_reduction(
        input, output, width, dispatch, kernel,
    ))
}

/// Prepare a ROCm rank-2 sum reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error.
pub fn prepare_sum_axis_into<'a, T>(
    device: &RocmDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    prepare_reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Prepare a ROCm rank-2 min reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error, including rejection of an
/// empty reduced axis.
pub fn prepare_min_axis_into<'a, T>(
    device: &RocmDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    prepare_reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Prepare a ROCm rank-2 max reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error, including rejection of an
/// empty reduced axis.
pub fn prepare_max_axis_into<'a, T>(
    device: &RocmDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    prepare_reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Prepare a ROCm rank-2 mean reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error, including rejection of an
/// empty reduced axis.
pub fn prepare_mean_axis_into<'a, T>(
    device: &RocmDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let Some(dispatch) = plan_dispatch(input, axis, output, width)? else {
        return Ok(empty_prepared_axis_reduction(input, output, width));
    };
    let kernel = cached_kernel(
        device,
        PipelineKey::MeanAxis {
            scalar: core::any::TypeId::of::<T>(),
            axis,
            width: width.get(),
        },
        "mean_axis_kernel",
        mean_axis_shader_source::<T>,
    )?;
    Ok(prepared_axis_reduction(
        input, output, width, dispatch, kernel,
    ))
}
