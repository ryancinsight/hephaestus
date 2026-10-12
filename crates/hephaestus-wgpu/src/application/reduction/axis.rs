use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DialectScalar, HephaestusError, IdentityToken, MaxOp,
    MinOp, OpIdentity, Result, SumOp, Wgsl,
};
use leto::Layout;

use super::prepared_axis::PreparedAxisReduction;
use super::prepared_axis::{
    axis_len, axis_reduction_pipeline, dispatch_axis_reduction, empty_prepared_axis_reduction,
    mean_axis_pipeline, plan_axis_reduction_dispatch, prepared_axis_reduction, reject_empty_axis,
};
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Prepare a rank-2 axis reduction over fixed input/output buffers and layouts.
///
/// # Errors
///
/// Returns a typed dispatch error when axis, shape, output-layout, width, or
/// aliasing validation fails.
pub fn prepare_reduce_axis_into<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let Some(mut dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(empty_prepared_axis_reduction());
    };
    let pipeline = axis_reduction_pipeline::<Op, T>(device, width, &mut dispatch);
    prepared_axis_reduction(
        device,
        pipeline,
        input,
        output,
        dispatch,
        "hephaestus-prepared-axis-reduction",
    )
}

/// Prepare a rank-2 axis sum over fixed input/output buffers and layouts.
#[inline]
pub fn prepare_sum_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    prepare_reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Prepare a rank-2 axis min over fixed input/output buffers and layouts.
#[inline]
pub fn prepare_min_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    prepare_reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Prepare a rank-2 axis max over fixed input/output buffers and layouts.
#[inline]
pub fn prepare_max_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    prepare_reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Prepare a rank-2 axis mean over fixed input/output buffers and layouts.
///
/// # Errors
///
/// Returns a typed dispatch error when the reduced axis is empty or when axis,
/// shape, output-layout, width, or aliasing validation fails.
pub fn prepare_mean_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    let reduced_axis_len = axis_len(input, axis)?;
    reject_empty_axis(reduced_axis_len, "mean_axis", axis)?;
    let Some(mut dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(empty_prepared_axis_reduction());
    };
    let pipeline = mean_axis_pipeline::<T>(device, width, &mut dispatch);
    prepared_axis_reduction(
        device,
        pipeline,
        input,
        output,
        dispatch,
        "hephaestus-prepared-mean-axis",
    )
}
/// Reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
///
/// This matches Leto's rank-preserving axis-reduction contract: an input with
/// shape `[rows, cols]` reduced over axis `0` writes shape `[1, cols]`, and
/// axis `1` writes shape `[rows, 1]`.
pub fn reduce_axis_into<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let Some(mut dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(());
    };
    let pipeline = axis_reduction_pipeline::<Op, T>(device, width, &mut dispatch);
    dispatch_axis_reduction(device, &pipeline, input, output, dispatch)
}

/// Reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
///
/// The output shape preserves the reduced axis as length one, matching
/// [`reduce_axis_into`].
pub fn reduce_axis<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        });
    }
    let mut output_shape = input.layout.shape();
    output_shape[axis] = 1;
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output_len = output_layout.checked_size().map_err(map_layout_err)?;
    let output = device.alloc_uninitialized::<T>(output_len)?;
    reduce_axis_into::<Op, T>(
        device,
        input,
        axis,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}
