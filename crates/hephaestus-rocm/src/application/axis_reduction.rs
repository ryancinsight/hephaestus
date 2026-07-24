//! ROCm rank-2 axis reductions over typed strided operands.

use crate::application::pipeline::{LaunchConfig, PipelineKey, cached_kernel, launch_kernel};
use crate::application::strided::StridedOperand;
use crate::infrastructure::DevicePtr;
use crate::{RocmBuffer, RocmDevice};
use bytemuck::Pod;
use hephaestus_core::{
    AxisReductionDispatch, AxisReductionMeta, CombineExpr, ComputeDevice, DeviceBuffer,
    DialectScalar, HephaestusError, HipC, IdentityToken, OpIdentity, Result, plan_axis_reduction,
};
use leto::Layout;

pub use hephaestus_core::{MaxOp, MinOp, SumOp};

fn axis_reduction_shader_source<Op: CombineExpr<HipC>, T: IdentityToken<Op, HipC>>() -> String {
    format!(
        r#"
struct AxisReductionMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

extern "C" __global__ void axis_reduction_kernel(
    AxisReductionMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta.offsets[3]) {{
        return;
    }}

    unsigned int axis = meta.offsets[2];
    unsigned int axis_len = (axis == 0u) ? meta.input_shape[0] : meta.input_shape[1];
    unsigned int out_row = (axis == 0u) ? 0u : i;
    unsigned int out_col = (axis == 0u) ? i : 0u;
    {ty} acc = {identity};

    for (unsigned int r = 0u; r < axis_len; r++) {{
        unsigned int in_row = (axis == 0u) ? r : out_row;
        unsigned int in_col = (axis == 0u) ? out_col : r;
        int in_off = (int)meta.offsets[0]
            + (int)in_row * meta.input_strides[0]
            + (int)in_col * meta.input_strides[1];
        {ty} lhs = acc;
        {ty} rhs = input[in_off];
        acc = {expr};
    }}

    int out_off = (int)meta.offsets[1]
        + (int)out_row * meta.output_strides[0]
        + (int)out_col * meta.output_strides[1];
    output[out_off] = acc;
}}
"#,
        ty = T::TYPE_TOKEN,
        identity = T::TOKEN,
        expr = Op::EXPR,
    )
}

fn mean_axis_shader_source<T: IdentityToken<SumOp, HipC>>() -> String {
    format!(
        r#"
struct AxisReductionMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

extern "C" __global__ void mean_axis_kernel(
    AxisReductionMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta.offsets[3]) {{
        return;
    }}

    unsigned int axis = meta.offsets[2];
    unsigned int axis_len = (axis == 0u) ? meta.input_shape[0] : meta.input_shape[1];
    unsigned int out_row = (axis == 0u) ? 0u : i;
    unsigned int out_col = (axis == 0u) ? i : 0u;
    {ty} acc = {identity};

    for (unsigned int r = 0u; r < axis_len; r++) {{
        unsigned int in_row = (axis == 0u) ? r : out_row;
        unsigned int in_col = (axis == 0u) ? out_col : r;
        int in_off = (int)meta.offsets[0]
            + (int)in_row * meta.input_strides[0]
            + (int)in_col * meta.input_strides[1];
        acc = acc + input[in_off];
    }}

    int out_off = (int)meta.offsets[1]
        + (int)out_row * meta.output_strides[0]
        + (int)out_col * meta.output_strides[1];
    output[out_off] = acc / ({ty})axis_len;
}}
"#,
        ty = T::TYPE_TOKEN,
        identity = T::TOKEN,
    )
}

fn axis_len<T>(input: StridedOperand<'_, T, 2>, axis: usize) -> Result<usize> {
    input
        .layout
        .shape
        .get(axis)
        .copied()
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        })
}

fn reject_empty_axis(axis_len: usize, operation: &'static str, axis: usize) -> Result<()> {
    if axis_len == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{operation} is undefined for empty axis {axis}"),
        });
    }
    Ok(())
}

fn plan_dispatch<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<Option<AxisReductionDispatch>>
where
    T: Pod,
{
    plan_axis_reduction(
        input.layout,
        input.buffer.len(),
        output.layout,
        output.buffer.len(),
        axis,
        width,
        input.buffer.aliases(output.buffer),
    )
}

fn launch_axis<Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    let Some(dispatch) = plan_dispatch(input, axis, output, width)? else {
        return Ok(());
    };

    let key = PipelineKey::AxisReduction {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        axis,
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "axis_reduction_kernel", || {
        axis_reduction_shader_source::<Op, T>()
    })?;
    launch_with_meta(
        device,
        &kernel,
        dispatch,
        input.buffer.raw(),
        output.buffer.raw(),
        width,
    )
}

fn launch_mean<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    let Some(dispatch) = plan_dispatch(input, axis, output, width)? else {
        return Ok(());
    };

    let key = PipelineKey::MeanAxis {
        scalar: core::any::TypeId::of::<T>(),
        axis,
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "mean_axis_kernel", || {
        mean_axis_shader_source::<T>()
    })?;
    launch_with_meta(
        device,
        &kernel,
        dispatch,
        input.buffer.raw(),
        output.buffer.raw(),
        width,
    )
}

fn launch_with_meta(
    device: &RocmDevice,
    kernel: &crate::application::pipeline::RocmKernel,
    dispatch: AxisReductionDispatch,
    input: DevicePtr,
    output: DevicePtr,
    width: hephaestus_core::BlockWidth,
) -> Result<()> {
    let mut meta = dispatch.meta;
    let mut input_ptr = input;
    let mut output_ptr = output;
    let mut args: [*mut core::ffi::c_void; 3] = [
        (&mut meta as *mut AxisReductionMeta).cast(),
        (&mut input_ptr as *mut DevicePtr).cast(),
        (&mut output_ptr as *mut DevicePtr).cast(),
    ];
    launch_kernel(
        device,
        kernel,
        LaunchConfig::linear(dispatch.groups, width),
        &mut args,
    )
}

/// Reduce a rank-2 strided operand along `axis` into caller-owned storage.
pub fn reduce_axis_into<Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    launch_axis::<Op, T>(device, input, axis, output, width)
}

/// Reduce a rank-2 strided operand along `axis` into a C-contiguous buffer.
pub fn reduce_axis<Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: hephaestus_core::BlockWidth,
) -> Result<RocmBuffer<T>>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        });
    }
    let mut output_shape = input.layout.shape;
    output_shape[axis] = 1;
    let output_layout =
        Layout::c_contiguous(output_shape).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("layout rejected: {error}"),
        })?;
    let output_len =
        output_layout
            .checked_size()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("layout rejected: {error}"),
            })?;
    let output = device.alloc_zeroed::<T>(output_len)?;
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

/// Sum-reduce a rank-2 operand along `axis` into caller-owned storage.
pub fn sum_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Sum-reduce a rank-2 operand along `axis` into a C-contiguous buffer.
pub fn sum_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: hephaestus_core::BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    reduce_axis::<SumOp, T>(device, input, axis, width)
}

/// Min-reduce a rank-2 operand along `axis` into caller-owned storage.
pub fn min_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Min-reduce a rank-2 operand along `axis` into a C-contiguous buffer.
pub fn min_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: hephaestus_core::BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis::<MinOp, T>(device, input, axis, width)
}

/// Max-reduce a rank-2 operand along `axis` into caller-owned storage.
pub fn max_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Max-reduce a rank-2 operand along `axis` into a C-contiguous buffer.
pub fn max_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: hephaestus_core::BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis::<MaxOp, T>(device, input, axis, width)
}

/// Mean-reduce a rank-2 operand along `axis` into caller-owned storage.
pub fn mean_axis_into<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: hephaestus_core::BlockWidth,
) -> Result<()>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    launch_mean(device, input, axis, output, width)
}

/// Mean-reduce a rank-2 operand along `axis` into a C-contiguous buffer.
pub fn mean_axis<T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: hephaestus_core::BlockWidth,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, HipC>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let mut output_shape = input.layout.shape;
    output_shape[axis] = 1;
    let output_layout =
        Layout::c_contiguous(output_shape).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("layout rejected: {error}"),
        })?;
    let output_len =
        output_layout
            .checked_size()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("layout rejected: {error}"),
            })?;
    let output = device.alloc_zeroed::<T>(output_len)?;
    mean_axis_into(
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
