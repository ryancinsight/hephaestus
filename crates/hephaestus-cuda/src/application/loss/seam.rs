use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    CrossEntropyPlan, Result, plan_cross_entropy_backward, plan_cross_entropy_forward,
};

use super::kernel::{
    backward_preflight_source, backward_source, forward_mean_source, forward_preflight_source,
    forward_source,
};
use super::metadata::{BackwardMeta, ForwardMeta};
use super::prepared::{
    PreparedBackwardSpec, PreparedCrossEntropyBackward, PreparedCrossEntropyForward,
    PreparedForwardSpec, compile,
};
use super::resources::{
    backward_aliases, forward_aliases, validate_backward_device, validate_forward_device,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// CUDA implementation of provider-owned mean cross-entropy.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaCrossEntropyOps;

impl CrossEntropyOps<CudaDevice, f32> for CudaCrossEntropyOps {
    type PreparedForward<'a> = PreparedCrossEntropyForward<'a>;
    type PreparedBackward<'a> = PreparedCrossEntropyBackward<'a>;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a CudaDevice,
        operands: CrossEntropyForwardOperands<'a, CudaBuffer<f32>, CudaBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        validate_forward_device(device, &operands)?;
        let plan = plan_cross_entropy_forward(&operands, forward_aliases(&operands))?;
        validate_plan(plan)?;
        let metadata = ForwardMeta::new(&operands)?;
        let preflight_kernel = compile(
            device,
            "cross_entropy_forward_preflight",
            forward_preflight_source,
        )?;
        let forward_kernel = compile(device, "cross_entropy_forward", forward_source)?;
        let mean_kernel = compile(device, "cross_entropy_forward_mean", forward_mean_source)?;
        let status = device.alloc_uninitialized::<u32>(1)?;
        let row_losses = device.alloc_uninitialized::<f32>(plan.batch)?;
        Ok(PreparedCrossEntropyForward::new(
            device,
            operands,
            PreparedForwardSpec {
                preflight_kernel,
                forward_kernel,
                mean_kernel,
                status,
                row_losses,
                metadata,
                batch: plan.batch,
            },
        ))
    }

    fn dispatch_cross_entropy_forward(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a CudaDevice,
        operands: CrossEntropyBackwardOperands<'a, CudaBuffer<f32>, CudaBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        validate_backward_device(device, &operands)?;
        let plan = plan_cross_entropy_backward(&operands, backward_aliases(&operands))?;
        validate_plan(plan)?;
        let metadata = BackwardMeta::new(&operands, plan.probability_tolerance)?;
        let preflight_kernel = compile(
            device,
            "cross_entropy_backward_preflight",
            backward_preflight_source,
        )?;
        let kernel = compile(device, "cross_entropy_backward", backward_source)?;
        let status = device.alloc_uninitialized::<u32>(1)?;
        Ok(PreparedCrossEntropyBackward::new(
            device,
            operands,
            PreparedBackwardSpec {
                preflight_kernel,
                kernel,
                status,
                metadata,
                batch: plan.batch,
                elements: plan.elements,
            },
        ))
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn validate_plan(plan: CrossEntropyPlan) -> Result<()> {
    plan.validate_address_limit(usize::try_from(i64::MAX).unwrap_or(usize::MAX))
}
