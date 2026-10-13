use hephaestus_core::{
    AxisReductionDispatch, BlockWidth, DeviceBuffer, HephaestusError, Result, plan_axis_reduction,
};

use crate::CudaDevice;
use crate::application::pipeline::{LaunchConfig, SafeCachedKernel, launch_kernel};
use crate::application::strided::StridedOperand;

pub(crate) fn axis_len<T>(input: StridedOperand<'_, T, 2>, axis: usize) -> Result<usize> {
    input
        .layout
        .shape()
        .get(axis)
        .copied()
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        })
}

pub(crate) fn reject_empty_axis(axis_len: usize, op_name: &'static str, axis: usize) -> Result<()> {
    if axis_len == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{op_name} is undefined for empty axis {axis}"),
        });
    }
    Ok(())
}

pub(crate) fn plan_axis_reduction_dispatch<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<Option<AxisReductionDispatch>> {
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

/// Launch a cached rank-2 axis-reduction kernel with a retained dispatch plan.
pub(crate) fn launch_axis_dispatch<T>(
    device: &CudaDevice,
    kernel: &SafeCachedKernel,
    dispatch: AxisReductionDispatch,
    input: &StridedOperand<'_, T, 2>,
    output: &StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()> {
    let mut meta_val = dispatch.meta;
    let mut in_ptr = input.buffer.raw();
    let mut out_ptr = output.buffer.raw();

    // Argument list mirrors `axis_reduction_kernel(AxisReductionMeta, const T*, T*)`.
    let mut args: [*mut std::ffi::c_void; 3] = [
        std::ptr::from_mut(&mut meta_val).cast(),
        std::ptr::from_mut(&mut in_ptr).cast(),
        std::ptr::from_mut(&mut out_ptr).cast(),
    ];

    launch_kernel(
        device,
        kernel,
        LaunchConfig::linear(dispatch.groups, width),
        &mut args,
    )
}
