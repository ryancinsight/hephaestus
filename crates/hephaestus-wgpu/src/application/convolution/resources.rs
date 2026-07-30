use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, HephaestusError, Result,
};

use super::metadata::ConvolutionMeta;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::{UniformBufferGuard, uniform_guard};

pub(super) fn metadata_buffer(
    device: &WgpuDevice,
    metadata: &ConvolutionMeta,
) -> Result<UniformBufferGuard> {
    let raw = device.get_uniform_buffer(WgpuDevice::byte_size::<ConvolutionMeta>(1)?)?;
    let buffer = uniform_guard(device.clone(), raw);
    device
        .queue()
        .write_buffer(&buffer, 0, bytemuck::bytes_of(metadata));
    Ok(buffer)
}

pub(super) fn binding<'a, T>(binding: u32, buffer: &'a WgpuBuffer<T>) -> wgpu::BindGroupEntry<'a> {
    raw_binding(binding, buffer.raw())
}

pub(super) fn raw_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(super) fn forward_aliases<T, const R: usize>(
    operands: &ConvolutionForwardOperands<'_, WgpuBuffer<T>, R>,
) -> bool {
    operands.output.buffer.aliases(operands.input.buffer)
        || operands.output.buffer.aliases(operands.weight.buffer)
        || operands
            .bias
            .is_some_and(|bias| operands.output.buffer.aliases(bias.buffer))
}

pub(super) fn backward_aliases<T, const R: usize>(
    operands: &ConvolutionBackwardOperands<'_, WgpuBuffer<T>, R>,
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

pub(super) fn layout_error(error: leto::LetoError) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: format!("convolution target layout rejected: {error}"),
    }
}
