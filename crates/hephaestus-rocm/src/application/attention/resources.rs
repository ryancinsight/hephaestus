use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, HephaestusError, Result,
};

use crate::{RocmBuffer, RocmDevice};

pub(super) fn validate_forward_device<T>(
    device: &RocmDevice,
    operands: &AttentionForwardOperands<'_, RocmBuffer<T>, T>,
) -> Result<()> {
    let matches = buffer_matches(device, operands.query.buffer)
        && buffer_matches(device, operands.key.buffer)
        && buffer_matches(device, operands.value.buffer)
        && buffer_matches(device, operands.output.buffer)
        && buffer_matches(device, operands.weights.buffer)
        && operands
            .mask
            .grouped_keep()
            .is_none_or(|mask| buffer_matches(device, mask.view().buffer));
    require_matching_device(matches)
}

pub(super) fn validate_backward_device<T>(
    device: &RocmDevice,
    operands: &AttentionBackwardOperands<'_, RocmBuffer<T>, T>,
) -> Result<()> {
    let matches = buffer_matches(device, operands.grad_output.buffer)
        && buffer_matches(device, operands.query.buffer)
        && buffer_matches(device, operands.key.buffer)
        && buffer_matches(device, operands.value.buffer)
        && buffer_matches(device, operands.weights.buffer)
        && operands
            .gradients
            .query
            .is_none_or(|view| buffer_matches(device, view.buffer))
        && operands
            .gradients
            .key
            .is_none_or(|view| buffer_matches(device, view.buffer))
        && operands
            .gradients
            .value
            .is_none_or(|view| buffer_matches(device, view.buffer));
    require_matching_device(matches)
}

pub(super) fn forward_aliases<T>(
    operands: &AttentionForwardOperands<'_, RocmBuffer<T>, T>,
) -> bool {
    let output = operands.output.buffer;
    let weights = operands.weights.buffer;
    let primary_reads = [
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
    ];
    output.aliases(weights)
        || primary_reads
            .into_iter()
            .any(|read| output.aliases(read) || weights.aliases(read))
        || operands.mask.grouped_keep().is_some_and(|mask| {
            output.aliases(mask.view().buffer) || weights.aliases(mask.view().buffer)
        })
}

pub(super) fn backward_aliases<T>(
    operands: &AttentionBackwardOperands<'_, RocmBuffer<T>, T>,
) -> bool {
    let reads = [
        operands.grad_output.buffer,
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
        operands.weights.buffer,
    ];
    let targets = [
        operands.gradients.query.map(|view| view.buffer),
        operands.gradients.key.map(|view| view.buffer),
        operands.gradients.value.map(|view| view.buffer),
    ];
    let target_read_alias = targets
        .into_iter()
        .flatten()
        .any(|target| reads.into_iter().any(|read| target.aliases(read)));
    let [query, key, value] = targets;
    let target_alias = query.is_some_and(|query| {
        key.is_some_and(|key| query.aliases(key)) || value.is_some_and(|value| query.aliases(value))
    }) || key.is_some_and(|key| value.is_some_and(|value| key.aliases(value)));
    target_read_alias || target_alias
}

pub(super) fn same_device(left: &RocmDevice, right: &RocmDevice) -> bool {
    left.same_context(right)
}

fn require_matching_device(matches: bool) -> Result<()> {
    if matches {
        Ok(())
    } else {
        Err(HephaestusError::InvalidConfiguration {
            message: "ROCm attention operands must belong to the dispatch device".to_string(),
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
