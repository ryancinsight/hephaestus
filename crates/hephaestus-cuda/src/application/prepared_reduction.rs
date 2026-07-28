//! Reusable scalar reduction plans for CUDA.

use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, IdentityToken,
    OpIdentity, Result, reduction_pass_count, validate_reduction_width,
};

use crate::CudaDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, SafeCachedKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::application::reduction::shader_source;
use crate::infrastructure::buffer::CudaBuffer;

struct PreparedPass {
    kernel: Arc<SafeCachedKernel>,
    groups: u32,
    input_len: u32,
}

pub(crate) struct PreparedReductionPlan<'a, T> {
    device: &'a CudaDevice,
    width: BlockWidth,
    passes: Vec<PreparedPass>,
    outputs: Vec<CudaBuffer<T>>,
}

/// A reusable CUDA scalar reduction plan over a fixed input buffer.
pub struct PreparedReduction<'a, T> {
    input: &'a CudaBuffer<T>,
    plan: PreparedReductionPlan<'a, T>,
}

impl<'a, T> PreparedReductionPlan<'a, T> {
    pub(crate) fn prepare<Op>(
        device: &'a CudaDevice,
        input_len: usize,
        width: BlockWidth,
    ) -> Result<Self>
    where
        Op: CombineExpr<CudaC>,
        T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
    {
        validate_reduction_width(width)?;
        if input_len == 0 {
            return Ok(Self {
                device,
                width,
                passes: Vec::new(),
                outputs: vec![device.upload(&[T::IDENTITY])?],
            });
        }

        let pass_count = reduction_pass_count(input_len, width).max(1);
        let mut passes = Vec::with_capacity(pass_count);
        let mut outputs = Vec::with_capacity(pass_count);
        let mut current_len = input_len;
        let kernel = cached_kernel(
            device,
            PipelineKey::Reduction {
                op: core::any::TypeId::of::<Op>(),
                scalar: core::any::TypeId::of::<T>(),
                width: width.get(),
            },
            "reduction_kernel",
            || shader_source::<Op, T>(width),
        )?;

        loop {
            let groups = grid_size(current_len, width)?;
            let output_len = current_len.div_ceil(width.get() as usize);
            let output = device.alloc_zeroed::<T>(output_len)?;
            passes.push(PreparedPass {
                kernel: Arc::clone(&kernel),
                groups,
                input_len: u32::try_from(current_len).map_err(|_| {
                    hephaestus_core::HephaestusError::DispatchFailed {
                        message: format!("CUDA reduction length {current_len} exceeds u32 range"),
                    }
                })?,
            });
            outputs.push(output);
            if output_len == 1 {
                break;
            }
            current_len = output_len;
        }

        Ok(Self {
            device,
            width,
            passes,
            outputs,
        })
    }

    pub(crate) fn dispatch(&self, input: &CudaBuffer<T>) -> Result<()> {
        for (index, pass) in self.passes.iter().enumerate() {
            let source = if index == 0 {
                input.raw()
            } else {
                self.outputs[index - 1].raw()
            };
            let output = self.outputs[index].raw();
            let mut input_ptr = source;
            let mut output_ptr = output;
            let mut input_len = pass.input_len;
            let mut args: [*mut core::ffi::c_void; 3] = [
                (&mut input_ptr as *mut u64).cast(),
                (&mut output_ptr as *mut u64).cast(),
                (&mut input_len as *mut u32).cast(),
            ];
            launch_kernel(
                self.device,
                &pass.kernel,
                LaunchConfig::linear(pass.groups, self.width),
                &mut args,
            )?;
        }
        Ok(())
    }

    pub(crate) fn output(&self) -> &CudaBuffer<T> {
        self.outputs
            .last()
            .expect("invariant: prepared reduction always owns an output")
    }

    pub(crate) fn into_output(mut self) -> CudaBuffer<T> {
        self.outputs
            .pop()
            .expect("invariant: prepared reduction always owns an output")
    }
}

impl<T> PreparedReduction<'_, T> {
    /// Dispatch the prepared reduction and reuse its device-resident outputs.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error when a native CUDA launch fails.
    pub fn dispatch(&self) -> Result<()> {
        self.plan.dispatch(self.input)
    }

    /// Return the one-element output buffer holding the latest result.
    #[must_use]
    pub fn output(&self) -> &CudaBuffer<T> {
        self.plan.output()
    }
}

/// Submit several prepared CUDA reductions in order without host materialization.
///
/// # Errors
///
/// Returns the first native launch error encountered.
pub fn submit_prepared_reduction_batch<T>(reductions: &[&PreparedReduction<'_, T>]) -> Result<()> {
    for reduction in reductions {
        reduction.dispatch()?;
    }
    Ok(())
}

/// Prepare a scalar reduction with a caller-selected power-of-two block width.
///
/// The complete partial-output tree is allocated once. Subsequent dispatches
/// reuse those buffers and read the current contents of `input`.
///
/// # Errors
///
/// Returns a typed error when the width, allocation, or native kernel
/// compilation contract is invalid.
pub fn prepare_reduction_with_width<'a, Op, T>(
    device: &'a CudaDevice,
    input: &'a CudaBuffer<T>,
    width: BlockWidth,
) -> Result<PreparedReduction<'a, T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    let plan = PreparedReductionPlan::prepare::<Op>(device, input.len(), width)?;
    Ok(PreparedReduction { input, plan })
}

/// Prepare a scalar reduction using the default block width.
#[inline]
pub fn prepare_reduction<'a, Op, T>(
    device: &'a CudaDevice,
    input: &'a CudaBuffer<T>,
) -> Result<PreparedReduction<'a, T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    prepare_reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}
