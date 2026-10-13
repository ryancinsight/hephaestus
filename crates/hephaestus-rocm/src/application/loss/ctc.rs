//! ROCm/HIP connectionist temporal classification.
//!
//! Monomorphization of the shared C-dialect CTC kernel over HIP
//! multi-storage kernels. The recurrences, the kernel source, and the
//! dispatch live in [`hephaestus_core::CtcKernel`]; this module pins the
//! ROCm substrate and binds the step traits to it.

use hephaestus_core::{CtcKernel as SharedCtcKernel, CtcStateBuffers, Result};
pub use hephaestus_core::{CtcOps, CtcStepOps};

use crate::RocmDevice;
use crate::application::storage_kernel::RocmMultiStorageKernel;
use crate::infrastructure::RocmBuffer;

/// Compiled HIP CTC step kernels.
///
/// Built once per device and reused across problems; the frame and the
/// seed ride in each launch's parameter block.
pub type CtcKernel = SharedCtcKernel<RocmMultiStorageKernel>;

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
