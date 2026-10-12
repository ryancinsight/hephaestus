use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    CrossEntropyScalar, Result, plan_cross_entropy_backward, plan_cross_entropy_forward,
};

use super::kernel::{
    BACKWARD_ENTRY, BACKWARD_PREFLIGHT_ENTRY, FORWARD_ENTRY, FORWARD_MEAN_ENTRY,
    FORWARD_PREFLIGHT_ENTRY, HipCrossEntropyScalar, backward_source, forward_source,
};
use super::metadata::CrossEntropyMeta;
use super::prepared::{PreparedRocmCrossEntropyBackward, PreparedRocmCrossEntropyForward, compile};
use super::resources::{
    backward_aliases, forward_aliases, validate_backward_device, validate_forward_device,
};
use crate::{RocmBuffer, RocmDevice};

/// ROCm implementation of provider-owned mean cross-entropy.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmCrossEntropyOps;

fn prepare_forward<'a, T: HipCrossEntropyScalar + CrossEntropyScalar>(
    device: &'a RocmDevice,
    operands: CrossEntropyForwardOperands<'a, RocmBuffer<T>, RocmBuffer<u32>>,
) -> Result<PreparedRocmCrossEntropyForward<'a, T>> {
    validate_forward_device(device, &operands)?;
    let plan = plan_cross_entropy_forward(&operands, forward_aliases(&operands))?;
    plan.validate_address_limit(
        usize::try_from(i32::MAX).expect("invariant: usize represents every i32 value"),
    )?;
    let metadata = CrossEntropyMeta::forward(&operands, plan, T::tolerance(&plan))?;
    let preflight = compile::<T>(device, FORWARD_PREFLIGHT_ENTRY, forward_source::<T>)?;
    let forward = compile::<T>(device, FORWARD_ENTRY, forward_source::<T>)?;
    let mean = compile::<T>(device, FORWARD_MEAN_ENTRY, forward_source::<T>)?;
    let status = device.alloc_zeroed::<u32>(1)?;
    let row_losses = device.alloc_uninitialized::<T>(plan.batch)?;
    Ok(PreparedRocmCrossEntropyForward::new(
        device,
        preflight,
        forward,
        mean,
        status,
        row_losses,
        operands.logits.buffer,
        operands.targets.buffer,
        operands.loss.buffer,
        operands.probabilities.buffer,
        metadata,
        plan.batch,
    ))
}

fn prepare_backward<'a, T: HipCrossEntropyScalar + CrossEntropyScalar>(
    device: &'a RocmDevice,
    operands: CrossEntropyBackwardOperands<'a, RocmBuffer<T>, RocmBuffer<u32>>,
) -> Result<PreparedRocmCrossEntropyBackward<'a, T>> {
    validate_backward_device(device, &operands)?;
    let plan = plan_cross_entropy_backward(&operands, backward_aliases(&operands))?;
    plan.validate_address_limit(
        usize::try_from(i32::MAX).expect("invariant: usize represents every i32 value"),
    )?;
    let metadata = CrossEntropyMeta::backward(&operands, plan, T::tolerance(&plan))?;
    let preflight = compile::<T>(device, BACKWARD_PREFLIGHT_ENTRY, backward_source::<T>)?;
    let backward = compile::<T>(device, BACKWARD_ENTRY, backward_source::<T>)?;
    let status = device.alloc_zeroed::<u32>(1)?;
    Ok(PreparedRocmCrossEntropyBackward::new(
        device,
        preflight,
        backward,
        status,
        operands.output_gradient.buffer,
        operands.probabilities.buffer,
        operands.targets.buffer,
        operands.logit_gradient.buffer,
        metadata,
        plan.batch,
        plan.elements,
    ))
}

impl CrossEntropyOps<RocmDevice, f32> for RocmCrossEntropyOps {
    type PreparedForward<'a>
        = PreparedRocmCrossEntropyForward<'a, f32>
    where
        RocmDevice: 'a;
    type PreparedBackward<'a>
        = PreparedRocmCrossEntropyBackward<'a, f32>
    where
        RocmDevice: 'a;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a RocmDevice,
        operands: CrossEntropyForwardOperands<'a, RocmBuffer<f32>, RocmBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        prepare_forward(device, operands)
    }

    fn dispatch_cross_entropy_forward(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a RocmDevice,
        operands: CrossEntropyBackwardOperands<'a, RocmBuffer<f32>, RocmBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        prepare_backward(device, operands)
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

impl CrossEntropyOps<RocmDevice, f64> for RocmCrossEntropyOps {
    type PreparedForward<'a>
        = PreparedRocmCrossEntropyForward<'a, f64>
    where
        RocmDevice: 'a;
    type PreparedBackward<'a>
        = PreparedRocmCrossEntropyBackward<'a, f64>
    where
        RocmDevice: 'a;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a RocmDevice,
        operands: CrossEntropyForwardOperands<'a, RocmBuffer<f64>, RocmBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        prepare_forward(device, operands)
    }

    fn dispatch_cross_entropy_forward(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a RocmDevice,
        operands: CrossEntropyBackwardOperands<'a, RocmBuffer<f64>, RocmBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        prepare_backward(device, operands)
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}
