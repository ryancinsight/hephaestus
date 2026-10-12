use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    CrossEntropyScalar, Result, plan_cross_entropy_backward, plan_cross_entropy_forward,
};

use super::kernel::{
    CudaCrossEntropyScalar, backward_preflight_source, backward_source, forward_mean_source,
    forward_preflight_source, forward_source,
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

/// Provider-owned CUDA implementation of mean cross-entropy.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaCrossEntropyOps;

fn prepare_forward<'a, T: CudaCrossEntropyScalar>(
    device: &'a CudaDevice,
    operands: CrossEntropyForwardOperands<'a, CudaBuffer<T>, CudaBuffer<u32>>,
) -> Result<PreparedCrossEntropyForward<'a, T>> {
    validate_forward_device(device, &operands)?;
    let plan = plan_cross_entropy_forward(&operands, forward_aliases(&operands))?;
    let metadata = ForwardMeta::new(&operands)?;
    let status = device.alloc_zeroed(1)?;
    let row_losses = device.alloc_zeroed(plan.batch)?;
    Ok(PreparedCrossEntropyForward::new(
        device,
        operands,
        PreparedForwardSpec {
            preflight_kernel: compile::<T>(
                device,
                "cross_entropy_forward_preflight",
                forward_preflight_source::<T>,
            )?,
            forward_kernel: compile::<T>(device, "cross_entropy_forward", forward_source::<T>)?,
            mean_kernel: compile::<T>(
                device,
                "cross_entropy_forward_mean",
                forward_mean_source::<T>,
            )?,
            status,
            row_losses,
            metadata,
            batch: plan.batch,
        },
    ))
}

fn prepare_backward<'a, T: CudaCrossEntropyScalar + CrossEntropyScalar>(
    device: &'a CudaDevice,
    operands: CrossEntropyBackwardOperands<'a, CudaBuffer<T>, CudaBuffer<u32>>,
) -> Result<PreparedCrossEntropyBackward<'a, T>> {
    validate_backward_device(device, &operands)?;
    let plan = plan_cross_entropy_backward(&operands, backward_aliases(&operands))?;
    let metadata = BackwardMeta::new(&operands, T::tolerance(&plan))?;
    let status = device.alloc_zeroed(1)?;
    Ok(PreparedCrossEntropyBackward::new(
        device,
        operands,
        PreparedBackwardSpec {
            preflight_kernel: compile::<T>(
                device,
                "cross_entropy_backward_preflight",
                backward_preflight_source::<T>,
            )?,
            kernel: compile::<T>(device, "cross_entropy_backward", backward_source::<T>)?,
            status,
            metadata,
            batch: plan.batch,
            elements: plan.elements,
        },
    ))
}

impl CrossEntropyOps<CudaDevice, f32> for CudaCrossEntropyOps {
    type PreparedForward<'a> = PreparedCrossEntropyForward<'a, f32>;
    type PreparedBackward<'a> = PreparedCrossEntropyBackward<'a, f32>;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a CudaDevice,
        operands: CrossEntropyForwardOperands<'a, CudaBuffer<f32>, CudaBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        prepare_forward(device, operands)
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
        prepare_backward(device, operands)
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

impl CrossEntropyOps<CudaDevice, f64> for CudaCrossEntropyOps {
    type PreparedForward<'a> = PreparedCrossEntropyForward<'a, f64>;
    type PreparedBackward<'a> = PreparedCrossEntropyBackward<'a, f64>;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a CudaDevice,
        operands: CrossEntropyForwardOperands<'a, CudaBuffer<f64>, CudaBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        prepare_forward(device, operands)
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
        operands: CrossEntropyBackwardOperands<'a, CudaBuffer<f64>, CudaBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        prepare_backward(device, operands)
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}
