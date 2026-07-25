//! Reusable rank-2 axis reduction plans for CUDA.

use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{
    AxisReductionDispatch, BlockWidth, CombineExpr, CudaC, DialectScalar, IdentityToken, MaxOp,
    MinOp, OpIdentity, Result, SumOp,
};

use crate::CudaDevice;
use crate::application::pipeline::{PipelineKey, SafeCachedKernel, cached_kernel};
use crate::application::reduction::{
    axis_len, axis_reduction_shader_source, launch_axis_dispatch, mean_axis_shader_source,
    plan_axis_reduction_dispatch, reject_empty_axis,
};
use crate::application::strided::StridedOperand;

/// A reusable CUDA rank-2 axis reduction over fixed input and output views.
pub struct PreparedAxisReduction<'a, T> {
    input: StridedOperand<'a, T, 2>,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
    dispatch: Option<AxisReductionDispatch>,
    kernel: Option<Arc<SafeCachedKernel>>,
}

impl<T> PreparedAxisReduction<'_, T> {
    /// Dispatch the prepared axis reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when the native CUDA launch fails.
    pub fn dispatch(&self, device: &CudaDevice) -> Result<()> {
        let (Some(dispatch), Some(kernel)) = (self.dispatch, self.kernel.as_ref()) else {
            return Ok(());
        };
        launch_axis_dispatch(
            device,
            kernel,
            dispatch,
            self.input,
            self.output,
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
    kernel: Arc<SafeCachedKernel>,
) -> PreparedAxisReduction<'a, T> {
    PreparedAxisReduction {
        input,
        output,
        width,
        dispatch: Some(dispatch),
        kernel: Some(kernel),
    }
}

/// Submit several prepared CUDA axis reductions in order.
///
/// # Errors
///
/// Returns the first native launch error encountered.
pub fn submit_prepared_axis_reduction_batch<T>(
    device: &CudaDevice,
    reductions: &[&PreparedAxisReduction<'_, T>],
) -> Result<()> {
    for reduction in reductions {
        reduction.dispatch(device)?;
    }
    Ok(())
}

/// Prepare a generic CUDA rank-2 axis reduction into fixed output storage.
///
/// # Errors
///
/// Returns a typed error when the shared axis planner rejects the axis, shape,
/// layout, width, aliasing, or buffer-size contract, or when kernel
/// compilation fails.
pub fn prepare_reduce_axis_into<'a, Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    let Some(dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
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

/// Prepare a CUDA rank-2 sum reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error.
pub fn prepare_sum_axis_into<'a, T>(
    device: &CudaDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    prepare_reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Prepare a CUDA rank-2 min reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error, including rejection of an
/// empty reduced axis.
pub fn prepare_min_axis_into<'a, T>(
    device: &CudaDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    prepare_reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Prepare a CUDA rank-2 max reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error, including rejection of an
/// empty reduced axis.
pub fn prepare_max_axis_into<'a, T>(
    device: &CudaDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    prepare_reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Prepare a CUDA rank-2 mean reduction along `axis` into fixed output storage.
///
/// # Errors
///
/// Returns a typed preparation or validation error, including rejection of an
/// empty reduced axis.
pub fn prepare_mean_axis_into<'a, T>(
    device: &CudaDevice,
    input: StridedOperand<'a, T, 2>,
    axis: usize,
    output: StridedOperand<'a, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<'a, T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let Some(dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
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
