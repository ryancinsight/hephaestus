//! Host-side planning and launch helpers for the ROCm reduction families.

use hephaestus_core::{
    AxisReductionDispatch, BlockWidth, DeviceBuffer, HephaestusError, Result, plan_axis_reduction,
};

use crate::RocmDevice;
use crate::application::pipeline::{LaunchConfig, RocmKernel, launch_kernel};
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

pub(crate) fn reject_empty_axis(
    axis_len: usize,
    operation: &'static str,
    axis: usize,
) -> Result<()> {
    if axis_len == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{operation} is undefined for empty axis {axis}"),
        });
    }
    Ok(())
}

pub(crate) fn plan_axis_reduction_dispatch<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<Option<AxisReductionDispatch>>
where
    T: eunomia::Pod,
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

/// Launch a cached rank-2 axis-reduction kernel with a retained dispatch plan.
pub(crate) fn launch_axis_dispatch<T>(
    device: &RocmDevice,
    kernel: &RocmKernel,
    dispatch: AxisReductionDispatch,
    input: &StridedOperand<'_, T, 2>,
    output: &StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()> {
    let mut meta = dispatch.meta;
    let mut input_ptr = input.buffer.raw();
    let mut output_ptr = output.buffer.raw();

    // Argument list mirrors `axis_reduction_kernel(AxisReductionMeta, const T*, T*)`.
    let mut args: [*mut core::ffi::c_void; 3] = [
        core::ptr::addr_of_mut!(meta)
            .cast::<hephaestus_core::AxisReductionMeta>()
            .cast(),
        core::ptr::addr_of_mut!(input_ptr)
            .cast::<crate::infrastructure::DevicePtr>()
            .cast(),
        core::ptr::addr_of_mut!(output_ptr)
            .cast::<crate::infrastructure::DevicePtr>()
            .cast(),
    ];
    launch_kernel(
        device,
        kernel,
        LaunchConfig::linear(dispatch.groups, width),
        &mut args,
    )
}
