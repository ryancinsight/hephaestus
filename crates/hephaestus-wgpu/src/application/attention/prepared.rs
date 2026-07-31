use hephaestus_core::Result;

use crate::application::pipeline::encode_compute_pass;
use crate::application::prepared::{checked_submit, device_owner, validate_device_owner};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};
use crate::infrastructure::pool::UniformBufferGuard;

pub(super) struct PreparedAttentionKernel {
    owner: PipelineCache,
    state: PreparedKernelState,
}

#[expect(
    clippy::large_enum_variant,
    reason = "boxing the ready state would allocate once more for every prepared kernel"
)]
enum PreparedKernelState {
    Empty,
    Ready {
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        _metadata: UniformBufferGuard,
        groups: u32,
        label: &'static str,
    },
}

impl PreparedAttentionKernel {
    pub(super) fn empty(device: &WgpuDevice) -> Self {
        Self {
            owner: device_owner(device),
            state: PreparedKernelState::Empty,
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

    fn validate_device(&self, device: &WgpuDevice) -> Result<()> {
        validate_device_owner(&self.owner, device, "attention")
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

    const fn label(&self) -> Option<&'static str> {
        match &self.state {
            PreparedKernelState::Empty => None,
            PreparedKernelState::Ready { label, .. } => Some(label),
        }
    }
}

/// A forward pass whose weight and output stages were prepared atomically.
pub struct PreparedAttentionForward {
    weights: PreparedAttentionKernel,
    output: PreparedAttentionKernel,
}

impl PreparedAttentionForward {
    pub(super) const fn new(
        weights: PreparedAttentionKernel,
        output: PreparedAttentionKernel,
    ) -> Self {
        Self { weights, output }
    }

    pub(super) fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        dispatch_kernels(device, [&self.weights, &self.output])
    }
}

/// Every selected additive-gradient stage plus its device-resident score workspace.
pub struct PreparedAttentionBackward {
    score: Option<PreparedAttentionKernel>,
    query: Option<PreparedAttentionKernel>,
    key: Option<PreparedAttentionKernel>,
    value: Option<PreparedAttentionKernel>,
    _score_workspace: Option<WgpuBuffer<f32>>,
}

impl PreparedAttentionBackward {
    pub(super) const fn new(
        score: Option<PreparedAttentionKernel>,
        query: Option<PreparedAttentionKernel>,
        key: Option<PreparedAttentionKernel>,
        value: Option<PreparedAttentionKernel>,
        score_workspace: Option<WgpuBuffer<f32>>,
    ) -> Self {
        Self {
            score,
            query,
            key,
            value,
            _score_workspace: score_workspace,
        }
    }

    pub(super) fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        dispatch_kernels(
            device,
            [
                self.score.as_ref(),
                self.query.as_ref(),
                self.key.as_ref(),
                self.value.as_ref(),
            ]
            .into_iter()
            .flatten(),
        )
    }
}

fn dispatch_kernels<'a>(
    device: &WgpuDevice,
    kernels: impl IntoIterator<Item = &'a PreparedAttentionKernel>,
) -> Result<()> {
    let kernels: smallvec::SmallVec<[&PreparedAttentionKernel; 4]> = kernels.into_iter().collect();
    for kernel in &kernels {
        kernel.validate_device(device)?;
    }
    let Some(label) = kernels.iter().find_map(|kernel| kernel.label()) else {
        return Ok(());
    };
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    for kernel in kernels {
        kernel.encode(&mut encoder);
    }
    checked_submit(device, label, encoder)
}
