use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    IdentityToken, OpIdentity, Result,
};
use leto::Layout;

use crate::CudaDevice;
use crate::application::pipeline::{PipelineKey, cached_kernel};
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;

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
    // already ignores strides on unit extents, so consumer spellings such
    // as left-padded `[1, n]` with strides `[0, 1]` qualify.
    scalar_output && layout.is_c_contiguous()
}

/// Whether a rank-2 axis reduction collapses to a scalar over a dense
/// full-buffer input — exactly a full reduction.
///
/// The axis kernel spawns one thread per output with serial per-thread
/// accumulation, so a scalar reduction over N elements runs on ONE thread
/// (~13.5ms at N=1M, measured). The parallel full reduction does the same
/// work in ~120us end-to-end. The dense full-buffer conditions below are
/// what make the two spellings identical element-for-element.
fn is_dense_scalar_reduction<T>(input: &StridedOperand<'_, T, 2>, axis: usize) -> bool {
    let shape = input.layout.shape();
    // Non-empty: on an empty axis the kernel path writes the identity while
    // the full reduction returns an empty buffer, so empty stays slow.
    is_dense_scalar_layout(input.layout, axis)
        && !input.buffer.is_empty()
        && input.buffer.len() == shape[0] * shape[1]
}

/// Reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
///
/// A scalar reduction over a dense full-buffer input dispatches to the
/// parallel full reduction instead of the one-thread-per-output axis kernel;
/// all other shapes take the axis path.
pub fn reduce_axis_into<Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    let Some(dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(());
    };

    if is_dense_scalar_reduction(&input, axis) {
        let full = reduction::<Op, T>(device, input.buffer)?;
        device.copy_buffer(&full, output.buffer)?;
        return Ok(());
    }

    let key = PipelineKey::AxisReduction {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        axis,
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "axis_reduction_kernel", || {
        axis_reduction_shader_source::<Op, T>()
    })?;

    launch_axis_dispatch(device, &kernel, dispatch, &input, &output, width)
}

/// Reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn reduce_axis<Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
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
