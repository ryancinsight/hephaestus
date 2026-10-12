use hephaestus_core::{CtcOps, CtcStateBuffers, CtcStepOps, Result};

use super::kernel::CtcKernel;
use crate::RocmDevice;
use crate::infrastructure::RocmBuffer;

/// Provider-owned implementation of [`CtcStepOps`] and [`CtcOps`] for ROCm.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmCtcOps;

impl CtcStepOps<RocmDevice> for RocmCtcOps {
    type CtcStep = CtcKernel;

    fn prepare_ctc_step(&self, device: &RocmDevice) -> Result<Self::CtcStep> {
        CtcKernel::new(device)
    }

    fn ctc_alpha_step_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<()> {
        kernel.alpha_step(device, t, log_probs, state)
    }

    fn ctc_beta_step_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<()> {
        kernel.beta_step(device, t, log_probs, state)
    }

    fn ctc_loss_finish_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::CtcStep,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<()> {
        kernel.loss_finish(device, state)
    }

    fn ctc_posterior_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::CtcStep,
        upstream: f32,
        grad: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}

impl CtcOps<RocmDevice> for RocmCtcOps {
    type Ctc = CtcKernel;

    fn prepare_ctc(&self, device: &RocmDevice) -> Result<Self::Ctc> {
        CtcKernel::new(device)
    }

    fn ctc_forward_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::Ctc,
        log_probs: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<f32> {
        hephaestus_core::drive_ctc_forward(self, device, kernel, log_probs, state)
    }

    fn ctc_backward_into(
        &self,
        device: &RocmDevice,
        kernel: &Self::Ctc,
        upstream: f32,
        grad: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}
