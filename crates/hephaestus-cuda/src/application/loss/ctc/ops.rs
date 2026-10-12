use hephaestus_core::{CtcOps, CtcStateBuffers, CtcStepOps, Result};

use super::kernel::CtcKernel;
use crate::CudaDevice;
use crate::infrastructure::buffer::CudaBuffer;

/// Provider-owned implementation of [`CtcStepOps`] and [`CtcOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaCtcOps;

impl CtcStepOps<CudaDevice> for CudaCtcOps {
    type CtcStep = CtcKernel;

    fn prepare_ctc_step(&self, device: &CudaDevice) -> Result<Self::CtcStep> {
        CtcKernel::new(device)
    }

    fn ctc_alpha_step_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.alpha_step(device, t, log_probs, state)
    }

    fn ctc_beta_step_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.beta_step(device, t, log_probs, state)
    }

    fn ctc_loss_finish_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.loss_finish(device, state)
    }

    fn ctc_posterior_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        upstream: f32,
        grad: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}

impl CtcOps<CudaDevice> for CudaCtcOps {
    type Ctc = CtcKernel;

    fn prepare_ctc(&self, device: &CudaDevice) -> Result<Self::Ctc> {
        CtcKernel::new(device)
    }

    fn ctc_forward_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::Ctc,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<f32> {
        hephaestus_core::drive_ctc_forward(self, device, kernel, log_probs, state)
    }

    fn ctc_backward_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::Ctc,
        upstream: f32,
        grad: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}
