use hephaestus_core::{ComputeDevice, Result};

use super::resources::semantic_error;
use crate::application::pipeline::encode_compute_pass;
use crate::application::prepared::{checked_submit, device_owner, validate_device_owner};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};
use crate::infrastructure::pool::UniformBufferGuard;

pub(super) struct PreparedCrossEntropyKernel {
    owner: PipelineCache,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _metadata: UniformBufferGuard,
    groups: u32,
    label: &'static str,
}

impl PreparedCrossEntropyKernel {
    pub(super) fn new(
        device: &WgpuDevice,
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        metadata: UniformBufferGuard,
        groups: u32,
        label: &'static str,
    ) -> Self {
        Self {
            owner: device_owner(device),
            pipeline,
            bind_group,
            _metadata: metadata,
            groups,
            label,
        }
    }

    fn validate_device(&self, device: &WgpuDevice) -> Result<()> {
        validate_device_owner(&self.owner, device, "cross-entropy")
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        encode_compute_pass(
            encoder,
            &self.pipeline,
            &self.bind_group,
            self.groups,
            self.label,
        );
    }
}

/// Prepared WGPU forward cross-entropy resources.
pub struct PreparedCrossEntropyForward {
    preflight: PreparedCrossEntropyKernel,
    status: WgpuBuffer<u32>,
    probabilities: PreparedCrossEntropyKernel,
    mean: PreparedCrossEntropyKernel,
    _row_losses: WgpuBuffer<f32>,
}

impl PreparedCrossEntropyForward {
    pub(super) fn new(
        preflight: PreparedCrossEntropyKernel,
        status: WgpuBuffer<u32>,
        probabilities: PreparedCrossEntropyKernel,
        mean: PreparedCrossEntropyKernel,
        row_losses: WgpuBuffer<f32>,
    ) -> Self {
        Self {
            preflight,
            status,
            probabilities,
            mean,
            _row_losses: row_losses,
        }
    }

    pub(super) fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        dispatch_preflight(device, &self.preflight, &self.status)?;
        dispatch_kernels(device, [&self.probabilities, &self.mean])
    }
}

/// Prepared WGPU additive backward cross-entropy resources.
pub struct PreparedCrossEntropyBackward {
    row_preflight: PreparedCrossEntropyKernel,
    arithmetic_preflight: PreparedCrossEntropyKernel,
    status: WgpuBuffer<u32>,
    backward: PreparedCrossEntropyKernel,
}

impl PreparedCrossEntropyBackward {
    pub(super) fn new(
        row_preflight: PreparedCrossEntropyKernel,
        arithmetic_preflight: PreparedCrossEntropyKernel,
        status: WgpuBuffer<u32>,
        backward: PreparedCrossEntropyKernel,
    ) -> Self {
        Self {
            row_preflight,
            arithmetic_preflight,
            status,
            backward,
        }
    }

    pub(super) fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        for kernel in [&self.row_preflight, &self.arithmetic_preflight] {
            kernel.validate_device(device)?;
        }
        reset_status(device, &self.status);
        dispatch_kernels(device, [&self.row_preflight, &self.arithmetic_preflight])?;
        check_status(device, &self.status)?;
        dispatch_kernels(device, [&self.backward])
    }
}

fn dispatch_preflight(
    device: &WgpuDevice,
    preflight: &PreparedCrossEntropyKernel,
    status: &WgpuBuffer<u32>,
) -> Result<()> {
    preflight.validate_device(device)?;
    reset_status(device, status);
    dispatch_kernels(device, [preflight])?;
    check_status(device, status)
}

fn reset_status(device: &WgpuDevice, status: &WgpuBuffer<u32>) {
    device
        .queue()
        .write_buffer(status.raw(), 0, eunomia::layout::bytes_of(&u32::MAX));
}

fn check_status(device: &WgpuDevice, status: &WgpuBuffer<u32>) -> Result<()> {
    let code = device.download_owned(status)?;
    let status = *code
        .first()
        .expect("invariant: cross-entropy status buffer is nonempty");
    semantic_error(if status == u32::MAX { 0 } else { status })
}

fn dispatch_kernels<'a>(
    device: &WgpuDevice,
    kernels: impl IntoIterator<Item = &'a PreparedCrossEntropyKernel>,
) -> Result<()> {
    let kernels: smallvec::SmallVec<[&PreparedCrossEntropyKernel; 3]> =
        kernels.into_iter().collect();
    for kernel in &kernels {
        kernel.validate_device(device)?;
    }
    let label = kernels
        .first()
        .expect("invariant: cross-entropy dispatch contains a kernel")
        .label;
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    for kernel in kernels {
        kernel.encode(&mut encoder);
    }
    checked_submit(device, label, encoder)
}
