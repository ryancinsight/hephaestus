use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DialectScalar, HephaestusError, IdentityToken,
    OpIdentity, Result,
};
use leto::Layout;

use crate::CudaDevice;
use crate::application::pipeline::{PipelineKey, cached_kernel};
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;

use super::plan::{launch_axis_dispatch, plan_axis_reduction_dispatch};
use super::shaders::axis_reduction_shader_source;

/// Reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
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
