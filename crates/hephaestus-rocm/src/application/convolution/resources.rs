use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, HephaestusError, Result,
};

use crate::infrastructure::RocmBuffer;
use crate::infrastructure::device::RocmDevice;

pub(super) fn validate_forward_device<T, const R: usize>(
    device: &RocmDevice,
    operands: &ConvolutionForwardOperands<'_, RocmBuffer<T>, R>,
) -> Result<()> {
    let all_match = buffer_matches(device, operands.input.buffer)
        && buffer_matches(device, operands.weight.buffer)
        && operands
            .bias
            .is_none_or(|bias| buffer_matches(device, bias.buffer))
        && buffer_matches(device, operands.output.buffer);
    require_matching_device(all_match)
}

pub(super) fn validate_backward_device<T, const R: usize>(
    device: &RocmDevice,
    operands: &ConvolutionBackwardOperands<'_, RocmBuffer<T>, R>,
) -> Result<()> {
    let all_match = buffer_matches(device, operands.input.buffer)
        && buffer_matches(device, operands.weight.buffer)
        && buffer_matches(device, operands.grad_output.buffer)
        && operands
            .gradients
            .input
            .is_none_or(|view| buffer_matches(device, view.buffer))
        && operands
            .gradients
            .weight
            .is_none_or(|view| buffer_matches(device, view.buffer))
        && operands
            .gradients
            .bias
            .is_none_or(|view| buffer_matches(device, view.buffer));
    require_matching_device(all_match)
}

pub(super) fn forward_aliases<T, const R: usize>(
    operands: &ConvolutionForwardOperands<'_, RocmBuffer<T>, R>,
) -> bool {
    operands.output.buffer.aliases(operands.input.buffer)
        || operands.output.buffer.aliases(operands.weight.buffer)
        || operands
            .bias
            .is_some_and(|bias| operands.output.buffer.aliases(bias.buffer))
}

pub(super) fn backward_aliases<T, const R: usize>(
    operands: &ConvolutionBackwardOperands<'_, RocmBuffer<T>, R>,
) -> bool {
    let reads = [
        operands.input.buffer,
        operands.weight.buffer,
        operands.grad_output.buffer,
    ];
    let input = operands.gradients.input.map(|view| view.buffer);
    let weight = operands.gradients.weight.map(|view| view.buffer);
    let bias = operands.gradients.bias.map(|view| view.buffer);
    let target_reads_alias = [input, weight, bias]
        .into_iter()
        .flatten()
        .any(|target| reads.into_iter().any(|read| target.aliases(read)));
    let targets_alias = input.is_some_and(|input| {
        weight.is_some_and(|weight| input.aliases(weight))
            || bias.is_some_and(|bias| input.aliases(bias))
    }) || weight
        .is_some_and(|weight| bias.is_some_and(|bias| weight.aliases(bias)));
    target_reads_alias || targets_alias
}

fn require_matching_device(all_match: bool) -> Result<()> {
    if all_match {
        Ok(())
    } else {
        Err(HephaestusError::InvalidConfiguration {
            message: "ROCm convolution operands must belong to the dispatch device".to_string(),
        })
    }
}

pub(super) fn same_device(left: &RocmDevice, right: &RocmDevice) -> bool {
    left.same_context(right)
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn buffer_matches<T>(device: &RocmDevice, buffer: &RocmBuffer<T>) -> bool {
    std::sync::Arc::ptr_eq(&buffer.context, &device.context)
}

#[cfg(not(all(feature = "rocm", target_os = "linux")))]
fn buffer_matches<T>(_device: &RocmDevice, _buffer: &RocmBuffer<T>) -> bool {
    true
}
