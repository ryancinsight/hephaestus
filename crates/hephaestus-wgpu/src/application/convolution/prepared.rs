use hephaestus_core::Result;

use crate::application::pipeline::encode_compute_pass;
use crate::application::prepared::{checked_submit, device_owner, validate_device_owner};
use crate::infrastructure::device::{PipelineCache, WgpuDevice};
use crate::infrastructure::pool::UniformBufferGuard;

/// One compiled convolution kernel bound to its operands and metadata.
pub struct PreparedConvolutionKernel {
    owner: PipelineCache,
    state: PreparedKernelState,
}

#[expect(
    clippy::large_enum_variant,
    reason = "boxing the common ready state would add a heap allocation to every kernel preparation"
)]
enum PreparedKernelState {
    Empty {
        label: &'static str,
    },
    Ready {
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        _metadata: UniformBufferGuard,
        groups: u32,
        label: &'static str,
    },
}

impl PreparedConvolutionKernel {
    pub(super) fn empty(device: &WgpuDevice, label: &'static str) -> Self {
        Self {
            owner: device_owner(device),
            state: PreparedKernelState::Empty { label },
        }
    }

    pub(super) fn ready(
        device: &WgpuDevice,
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        metadata: UniformBufferGuard,
        groups: u32,
        label: &'static str,
    ) -> Self {
        Self {
            owner: device_owner(device),
            state: PreparedKernelState::Ready {
                pipeline,
                bind_group,
                _metadata: metadata,
                groups,
                label,
            },
        }
    }

    pub(super) fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        self.validate_device(device)?;
        if !self.is_ready() {
            return Ok(());
        }
        let label = self.label();
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        self.encode(&mut encoder);
        checked_submit(device, label, encoder)
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let PreparedKernelState::Ready {
            pipeline,
            bind_group,
            groups,
            label,
            ..
        } = &self.state
        else {
            return;
        };
        encode_compute_pass(encoder, pipeline, bind_group, *groups, label);
    }

    const fn label(&self) -> &'static str {
        match &self.state {
            PreparedKernelState::Empty { label } | PreparedKernelState::Ready { label, .. } => {
                label
            }
        }
    }

    const fn is_ready(&self) -> bool {
        matches!(&self.state, PreparedKernelState::Ready { .. })
    }

    fn validate_device(&self, device: &WgpuDevice) -> Result<()> {
        validate_device_owner(&self.owner, device, "convolution")
    }
}

/// Every selected additive-gradient kernel prepared as one atomic unit.
pub struct PreparedConvolutionBackward {
    pub(super) input: Option<PreparedConvolutionKernel>,
    pub(super) weight: Option<PreparedConvolutionKernel>,
    pub(super) bias: Option<PreparedConvolutionKernel>,
}

impl PreparedConvolutionBackward {
    pub(super) fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        let kernels = [&self.input, &self.weight, &self.bias];
        for prepared in kernels.into_iter().filter_map(|prepared| prepared.as_ref()) {
            prepared.validate_device(device)?;
        }
        let mut kernels = kernels
            .into_iter()
            .filter_map(|prepared| prepared.as_ref())
            .filter(|prepared| prepared.is_ready());
        let Some(first) = kernels.next() else {
            return Ok(());
        };
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(first.label()),
            });
        first.encode(&mut encoder);
        for prepared in kernels {
            prepared.encode(&mut encoder);
        }
        checked_submit(device, first.label(), encoder)
    }
}
