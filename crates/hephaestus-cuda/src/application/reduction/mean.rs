use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DialectScalar, IdentityToken, OpIdentity, Result, SumOp,
};
use leto::Layout;

use crate::CudaDevice;
use crate::application::pipeline::{PipelineKey, cached_kernel};
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::CudaBuffer;

use super::plan::{
    axis_len, launch_axis_dispatch, plan_axis_reduction_dispatch, reject_empty_axis,
};
use super::shaders::mean_axis_shader_source;

/// Mean-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
pub fn mean_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let Some(dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(());
    };

    let key = PipelineKey::MeanAxis {
        scalar: std::any::TypeId::of::<T>(),
        axis,
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "mean_axis_kernel", || {
        mean_axis_shader_source::<T>()
    })?;

    launch_axis_dispatch(device, &kernel, dispatch, &input, &output, width)
}

/// Mean-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn mean_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let mut output_shape = input.layout.shape();
    output_shape[axis] = 1;
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output_len = output_layout.checked_size().map_err(map_layout_err)?;
    let output = device.alloc_uninitialized::<T>(output_len)?;
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
