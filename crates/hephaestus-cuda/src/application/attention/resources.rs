use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, HephaestusError, Result,
};

use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

pub(super) fn validate_forward_device<T>(
    device: &CudaDevice,
    operands: &AttentionForwardOperands<'_, CudaBuffer<T>, T>,
) -> Result<()> {
    let matches = buffer_matches(device, operands.query.buffer)
        && buffer_matches(device, operands.key.buffer)
        && buffer_matches(device, operands.value.buffer)
        && buffer_matches(device, operands.output.buffer)
        && buffer_matches(device, operands.weights.buffer)
        && operands
            .mask
            .grouped_keep()
            .is_none_or(|keep| buffer_matches(device, keep.view().buffer));
    require_matching_device(matches)
}

pub(super) fn validate_backward_device<T>(
    device: &CudaDevice,
    operands: &AttentionBackwardOperands<'_, CudaBuffer<T>, T>,
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
    operands: &AttentionForwardOperands<'_, CudaBuffer<T>, T>,
) -> bool {
    let output = operands.output.buffer;
    let weights = operands.weights.buffer;
    let read_alias = [
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
    ]
    .into_iter()
    .any(|read| output.aliases(read) || weights.aliases(read));
    let mask_alias = operands.mask.grouped_keep().is_some_and(|keep| {
        output.aliases(keep.view().buffer) || weights.aliases(keep.view().buffer)
    });
    read_alias || mask_alias || output.aliases(weights)
}

pub(super) fn backward_aliases<T>(
    operands: &AttentionBackwardOperands<'_, CudaBuffer<T>, T>,
) -> bool {
    let reads = [
        operands.grad_output.buffer,
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
        operands.weights.buffer,
    ];
    let query = operands.gradients.query.map(|view| view.buffer);
    let key = operands.gradients.key.map(|view| view.buffer);
    let value = operands.gradients.value.map(|view| view.buffer);
    let aliases_read = [query, key, value]
        .into_iter()
        .flatten()
        .any(|target| reads.into_iter().any(|read| target.aliases(read)));
    let aliases_target = query.is_some_and(|query| {
        key.is_some_and(|key| query.aliases(key)) || value.is_some_and(|value| query.aliases(value))
    }) || key.is_some_and(|key| value.is_some_and(|value| key.aliases(value)));
    aliases_read || aliases_target
}

pub(super) fn same_device(left: &CudaDevice, right: &CudaDevice) -> bool {
    #[cfg(feature = "cuda")]
    {
        std::sync::Arc::ptr_eq(left.cuda_context(), right.cuda_context())
    }
    #[cfg(not(feature = "cuda"))]
    {
        std::ptr::eq(left, right)
    }
}

fn require_matching_device(matches: bool) -> Result<()> {
    if matches {
        Ok(())
    } else {
        Err(HephaestusError::InvalidConfiguration {
            message: "CUDA attention operands must belong to the dispatch device".to_string(),
        })
    }
}

#[cfg(feature = "cuda")]
fn buffer_matches<T>(device: &CudaDevice, buffer: &CudaBuffer<T>) -> bool {
    buffer
        .context
        .as_ref()
        .is_some_and(|context| std::sync::Arc::ptr_eq(context, device.cuda_context()))
}

#[cfg(not(feature = "cuda"))]
fn buffer_matches<T>(_device: &CudaDevice, _buffer: &CudaBuffer<T>) -> bool {
    true
}
