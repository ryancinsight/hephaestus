//! Command-stream recording and bound dispatch.

use crate::application::bindings::UniformBuffers;
use crate::application::prepared::checked_submit_with_timeout;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{DispatchGrid, Result};

/// WGPU command stream for ordered kernel dispatch, copies, and fills.
pub struct WgpuCommandStream<'d> {
    pub(crate) device: &'d WgpuDevice,
    pub(crate) encoder: wgpu::CommandEncoder,
    pub(crate) uniform_buffers: UniformBuffers,
}

/// A fully bound WGPU dispatch whose immutable resources are prepared once.
///
/// Device provenance is held and checked once by the enclosing prepared
/// operation before this crate-private command is encoded.
pub(crate) struct WgpuBoundDispatch {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) _parameters: wgpu::Buffer,
    pub(crate) grid: DispatchGrid,
}

impl WgpuBoundDispatch {
    pub(crate) fn encode_in_pass(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.grid.x == 0 || self.grid.y == 0 || self.grid.z == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.x, self.grid.y, self.grid.z);
    }
}

impl WgpuCommandStream<'_> {
    pub(crate) const fn device(&self) -> &WgpuDevice {
        self.device
    }

    pub(crate) fn encode_bound_sequence(
        &mut self,
        label: &str,
        prepared: &[WgpuBoundDispatch],
    ) -> Result<()> {
        let mut pass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        for command in prepared {
            command.encode_in_pass(&mut pass);
        }
        Ok(())
    }

    /// Submit this stream and wait at most `timeout` for device completion.
    ///
    /// This bounded form is intended for hosts and measurement harnesses whose
    /// external waits must have an explicit deadline.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::DispatchFailed`](hephaestus_core::HephaestusError::DispatchFailed)
    /// when submission, device
    /// completion, or a scoped WGPU error fails before the deadline.
    pub fn submit_with_timeout(self, timeout: std::time::Duration) -> Result<()> {
        checked_submit_with_timeout(
            self.device,
            "hephaestus-command-stream",
            self.encoder,
            Some(timeout),
        )?;
        drop(self.uniform_buffers);
        Ok(())
    }

    /// Submit this stream and return the submission's queue index, so the
    /// caller can wait on exactly this submission (index-scoped poll)
    /// instead of draining the whole queue.
    pub(crate) fn submit_indexed(self) -> Result<wgpu::SubmissionIndex> {
        let submission_index = checked_submit_with_timeout(
            self.device,
            "hephaestus-command-stream",
            self.encoder,
            None,
        )?;
        drop(self.uniform_buffers);
        Ok(submission_index)
    }
}
