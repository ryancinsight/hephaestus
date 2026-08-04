use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, HephaestusError, Result,
};

use crate::{RocmBuffer, RocmDevice};

pub(super) fn validate_forward_device(
    device: &RocmDevice,
    operands: &CrossEntropyForwardOperands<'_, RocmBuffer<f32>, RocmBuffer<u32>>,
) -> Result<()> {
    require_matching_device(
        buffer_matches(device, operands.logits.buffer)
            && buffer_matches(device, operands.targets.buffer)
            && buffer_matches(device, operands.loss.buffer)
            && buffer_matches(device, operands.probabilities.buffer),
    )
}

pub(super) fn validate_backward_device(
    device: &RocmDevice,
    operands: &CrossEntropyBackwardOperands<'_, RocmBuffer<f32>, RocmBuffer<u32>>,
) -> Result<()> {
    require_matching_device(
        buffer_matches(device, operands.output_gradient.buffer)
            && buffer_matches(device, operands.probabilities.buffer)
            && buffer_matches(device, operands.targets.buffer)
            && buffer_matches(device, operands.logit_gradient.buffer),
    )
}

pub(super) fn forward_aliases(
    operands: &CrossEntropyForwardOperands<'_, RocmBuffer<f32>, RocmBuffer<u32>>,
) -> bool {
    operands.loss.buffer.aliases(operands.logits.buffer)
        || operands.loss.buffer.aliases(operands.targets.buffer)
        || operands.loss.buffer.aliases(operands.probabilities.buffer)
        || operands
            .probabilities
            .buffer
            .aliases(operands.logits.buffer)
        || operands
            .probabilities
            .buffer
            .aliases(operands.targets.buffer)
}

pub(super) fn backward_aliases(
    operands: &CrossEntropyBackwardOperands<'_, RocmBuffer<f32>, RocmBuffer<u32>>,
) -> bool {
    operands
        .logit_gradient
        .buffer
        .aliases(operands.output_gradient.buffer)
        || operands
            .logit_gradient
            .buffer
            .aliases(operands.probabilities.buffer)
        || operands
            .logit_gradient
            .buffer
            .aliases(operands.targets.buffer)
}

pub(super) fn same_device(left: &RocmDevice, right: &RocmDevice) -> bool {
    left.same_context(right)
}

fn require_matching_device(matches: bool) -> Result<()> {
    if matches {
        Ok(())
    } else {
        Err(HephaestusError::InvalidConfiguration {
            message: "ROCm cross-entropy operands must belong to the dispatch device".to_string(),
        })
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn buffer_matches<T>(device: &RocmDevice, buffer: &RocmBuffer<T>) -> bool {
    std::sync::Arc::ptr_eq(&buffer.context, &device.context)
}

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
fn buffer_matches<T>(_device: &RocmDevice, _buffer: &RocmBuffer<T>) -> bool {
    true
}
