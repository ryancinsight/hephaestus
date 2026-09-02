use hephaestus_core::Result;

use crate::application::pipeline::encode_compute_pass;
use crate::application::prepared::{checked_submit, device_owner, validate_device_owner};
use crate::infrastructure::device::{PipelineCache, WgpuDevice};
use crate::infrastructure::pool::UniformBufferGuard;

/// One validated WGPU window kernel bound to its metadata and operands.
pub(super) struct PreparedWindowKernel {
    owner: PipelineCache,
    state: PreparedWindowState,
}

#[expect(
    clippy::large_enum_variant,
    reason = "boxing the common ready state would add an allocation to every preparation"
)]
enum PreparedWindowState {
    Empty,
    Ready {
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        _metadata: UniformBufferGuard,
        groups: u32,
        label: &'static str,
    },
}

impl PreparedWindowKernel {
    pub(super) fn empty(device: &WgpuDevice, _label: &'static str) -> Self {
        Self {
            owner: device_owner(device),
            state: PreparedWindowState::Empty,
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
            state: PreparedWindowState::Ready {
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
        let PreparedWindowState::Ready {
            pipeline,
            bind_group,
            groups,
            label,
            ..
        } = &self.state
        else {
            return Ok(());
        };
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encode_compute_pass(&mut encoder, pipeline, bind_group, *groups, label);
        checked_submit(device, label, encoder)
    }

    fn validate_device(&self, device: &WgpuDevice) -> Result<()> {
        validate_device_owner(&self.owner, device, "window")
    }
}
