//! Rank-2 axis reductions over strided operands on ROCm.

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC,
    IdentityToken, OpIdentity, Result, map_layout_err,
};
use leto::Layout;

use crate::RocmDevice;
use crate::application::pipeline::{PipelineKey, cached_kernel};
use crate::application::strided::StridedOperand;
use crate::infrastructure::RocmBuffer;

use super::full::reduction;
use super::plan::{launch_axis_dispatch, plan_axis_reduction_dispatch};
use super::shaders::axis_reduction_shader_source;

/// Whether a rank-2 layout is a dense scalar collapse along `axis` —
/// one output over a C-contiguous full-extent input.
///
/// Strides on unit extents are don't-care: a left-padded `[1, n]` spelling
/// with strides `[0, 1]` addresses the same dense block as canonical `[n, 1]`.
pub(crate) fn is_dense_scalar_layout(layout: &Layout<2>, axis: usize) -> bool {
    let shape = layout.shape();
    let scalar_output = (axis == 0 && shape[1] == 1) || (axis == 1 && shape[0] == 1);
    // Dense stride/offset judgment belongs to leto: `is_c_contiguous`
    // already ignores strides on unit extents and pins the offset to zero.
    scalar_output && layout.is_c_contiguous()
}

/// Whether a rank-2 axis reduction collapses to a scalar over a dense
/// full-buffer input — exactly a full reduction.
///
/// The axis kernel spawns one thread per output with serial per-thread
/// accumulation, so a scalar reduction over N elements runs on ONE thread.
/// The parallel full reduction does the same work across a whole workgroup
/// tree. The dense full-buffer conditions below are what make the two
/// spellings identical element-for-element.
fn is_dense_scalar_reduction<T>(input: &StridedOperand<'_, T, 2>, axis: usize) -> bool
where
    T: Pod,
{
    let shape = input.layout.shape();
    // Non-empty: on an empty axis the kernel path writes the identity while
    // the full reduction returns an empty buffer, so empty stays slow.
    is_dense_scalar_layout(input.layout, axis)
        && !input.buffer.is_empty()
        && input.buffer.len() == shape[0] * shape[1]
}

/// Whether the output view is the single element at the start of its buffer.
///
/// The dense-scalar fast path copies one reduced value into the output
/// buffer, so it is only exact when the output view starts at element zero
/// and owns the whole buffer.
fn is_scalar_output_at_origin<T>(output: &StridedOperand<'_, T, 2>) -> bool
where
    T: Pod,
{
    output.layout.offset() == 0 && output.buffer.len() == 1
}

/// Reduce a rank-2 strided operand along `axis` into caller-owned storage.
///
/// A scalar reduction over a dense full-buffer input dispatches to the
/// parallel full reduction instead of the one-thread-per-output axis kernel;
/// all other shapes take the axis path.
pub fn reduce_axis_into<Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    let Some(dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(());
    };

    if is_dense_scalar_reduction(&input, axis) && is_scalar_output_at_origin(&output) {
        let full = reduction::<Op, T>(device, input.buffer)?;
        device.copy_buffer(&full, output.buffer)?;
        return Ok(());
    }

    let key = PipelineKey::AxisReduction {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        axis,
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "axis_reduction_kernel", || {
        axis_reduction_shader_source::<Op, T>()
    })?;
    launch_axis_dispatch(device, &kernel, dispatch, &input, &output, width)
}

/// Reduce a rank-2 strided operand along `axis` into a C-contiguous buffer.
pub fn reduce_axis<Op, T>(
    device: &RocmDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
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
    if is_dense_scalar_reduction(&input, axis) {
        return reduction::<Op, T>(device, input.buffer);
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
