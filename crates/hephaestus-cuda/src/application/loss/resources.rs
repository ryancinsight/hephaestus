use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, HephaestusError, Result,
};

use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

pub(super) fn validate_forward_device(
    device: &CudaDevice,
    operands: &CrossEntropyForwardOperands<'_, CudaBuffer<f32>, CudaBuffer<u32>>,
) -> Result<()> {
    require_matching_device(
        buffer_matches(device, operands.logits.buffer)
            && buffer_matches(device, operands.targets.buffer)
            && buffer_matches(device, operands.loss.buffer)
            && buffer_matches(device, operands.probabilities.buffer),
    )
}

pub(super) fn validate_backward_device(
    device: &CudaDevice,
    operands: &CrossEntropyBackwardOperands<'_, CudaBuffer<f32>, CudaBuffer<u32>>,
) -> Result<()> {
    require_matching_device(
        buffer_matches(device, operands.output_gradient.buffer)
            && buffer_matches(device, operands.probabilities.buffer)
            && buffer_matches(device, operands.targets.buffer)
            && buffer_matches(device, operands.logit_gradient.buffer),
    )
}

pub(super) fn forward_aliases(
    operands: &CrossEntropyForwardOperands<'_, CudaBuffer<f32>, CudaBuffer<u32>>,
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
    operands: &CrossEntropyBackwardOperands<'_, CudaBuffer<f32>, CudaBuffer<u32>>,
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
            message: "CUDA cross-entropy operands must belong to the dispatch device".to_string(),
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
