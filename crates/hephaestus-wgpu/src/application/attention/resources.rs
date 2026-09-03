use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, HephaestusError, Result,
};

use super::metadata::AttentionMeta;
use crate::application::prepared::validate_buffer_owner;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::{UniformBufferGuard, uniform_guard};

pub(super) fn metadata_buffer(
    device: &WgpuDevice,
    metadata: &AttentionMeta,
) -> Result<UniformBufferGuard> {
    let raw = device.get_uniform_buffer(WgpuDevice::byte_size::<AttentionMeta>(1)?)?;
    let buffer = uniform_guard(device.clone(), raw);
    device
        .queue()
        .write_buffer(&buffer, 0, eunomia::layout::bytes_of(metadata));
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
    operands: &AttentionForwardOperands<'_, WgpuBuffer<f32>, f32>,
) -> Result<()> {
    for buffer in [
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
        operands.output.buffer,
        operands.weights.buffer,
    ] {
        validate_buffer_owner(buffer, device, "attention")?;
    }
    if let Some(keep) = operands.mask.grouped_keep() {
        validate_buffer_owner(keep.view().buffer, device, "attention")?;
    }
    Ok(())
}

pub(super) fn validate_backward_owners(
    device: &WgpuDevice,
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<f32>, f32>,
) -> Result<()> {
    for buffer in [
        operands.grad_output.buffer,
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
        operands.weights.buffer,
    ] {
        validate_buffer_owner(buffer, device, "attention")?;
    }
    for target in [
        operands.gradients.query,
        operands.gradients.key,
        operands.gradients.value,
    ]
    .into_iter()
    .flatten()
    {
        validate_buffer_owner(target.buffer, device, "attention")?;
    }
    Ok(())
}

pub(super) fn forward_aliases(
    operands: &AttentionForwardOperands<'_, WgpuBuffer<f32>, f32>,
) -> bool {
    let reads = [
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
    ];
    let mask = operands.mask.grouped_keep().map(|keep| keep.view().buffer);
    operands.output.buffer.aliases(operands.weights.buffer)
        || [operands.output.buffer, operands.weights.buffer]
            .into_iter()
            .any(|target| {
                reads.into_iter().any(|read| target.aliases(read))
                    || mask.is_some_and(|mask| target.aliases(mask))
            })
}

pub(super) fn backward_aliases(
    operands: &AttentionBackwardOperands<'_, WgpuBuffer<f32>, f32>,
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
    targets
        .into_iter()
        .flatten()
        .any(|target| reads.into_iter().any(|read| target.aliases(read)))
        || targets.iter().enumerate().any(|(index, target)| {
            target.is_some_and(|target| {
                targets[index + 1..]
                    .iter()
                    .flatten()
                    .any(|other| target.aliases(other))
            })
        })
}

pub(super) fn address_limit() -> usize {
    usize::try_from(i32::MAX).expect("invariant: supported WGPU hosts represent i32 in usize")
}

pub(super) fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}
