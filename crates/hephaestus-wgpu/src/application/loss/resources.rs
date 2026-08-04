use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyStatus, Result,
};

use super::metadata::CrossEntropyMeta;
use crate::application::prepared::validate_buffer_owner;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::{UniformBufferGuard, uniform_guard};

pub(super) fn metadata_buffer(
    device: &WgpuDevice,
    metadata: &CrossEntropyMeta,
) -> Result<UniformBufferGuard> {
    let raw = device.get_uniform_buffer(WgpuDevice::byte_size::<CrossEntropyMeta>(1)?)?;
    let buffer = uniform_guard(device.clone(), raw);
    device
        .queue()
        .write_buffer(&buffer, 0, bytemuck::bytes_of(metadata));
    Ok(buffer)
}

pub(super) fn binding<T>(binding: u32, buffer: &WgpuBuffer<T>) -> wgpu::BindGroupEntry<'_> {
    raw_binding(binding, buffer.raw())
}

pub(super) fn raw_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(super) fn validate_forward_owners(
    device: &WgpuDevice,
    operands: &CrossEntropyForwardOperands<'_, WgpuBuffer<f32>, WgpuBuffer<u32>>,
) -> Result<()> {
    validate_buffer_owner(operands.logits.buffer, device, "cross-entropy")?;
    validate_buffer_owner(operands.targets.buffer, device, "cross-entropy")?;
    validate_buffer_owner(operands.loss.buffer, device, "cross-entropy")?;
    validate_buffer_owner(operands.probabilities.buffer, device, "cross-entropy")
}

pub(super) fn validate_backward_owners(
    device: &WgpuDevice,
    operands: &CrossEntropyBackwardOperands<'_, WgpuBuffer<f32>, WgpuBuffer<u32>>,
) -> Result<()> {
    validate_buffer_owner(operands.output_gradient.buffer, device, "cross-entropy")?;
    validate_buffer_owner(operands.probabilities.buffer, device, "cross-entropy")?;
    validate_buffer_owner(operands.targets.buffer, device, "cross-entropy")?;
    validate_buffer_owner(operands.logit_gradient.buffer, device, "cross-entropy")
}

pub(super) fn forward_aliases(
    operands: &CrossEntropyForwardOperands<'_, WgpuBuffer<f32>, WgpuBuffer<u32>>,
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
    operands: &CrossEntropyBackwardOperands<'_, WgpuBuffer<f32>, WgpuBuffer<u32>>,
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

pub(super) fn address_limit() -> usize {
    usize::try_from(i32::MAX).expect("invariant: supported WGPU hosts represent i32 in usize")
}

pub(super) fn semantic_error(code: u32) -> Result<()> {
    CrossEntropyStatus::check(code)
}
