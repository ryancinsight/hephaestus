//! Metal CTC delegation through the WGPU backend.
//!
//! Metal owns no CTC kernels of its own; the entry points are the WGSL
//! ones, reached through the Metal-selected WGPU device. Each call
//! re-views the Metal state as WGPU state: the six buffers are handle
//! clones, and the problem metadata — kilobytes next to the grids —
//! is cloned per call rather than contorting the shared state type.

use hephaestus_core::{CtcOps, CtcStateBuffers, CtcStepOps, Result};
use hephaestus_wgpu as wgpu_backend;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

fn wgpu_state(state: &CtcStateBuffers<MetalDevice>) -> CtcStateBuffers<wgpu_backend::WgpuDevice> {
    CtcStateBuffers {
        problem: state.problem.clone(),
        labels: state.labels.wgpu_buffer().clone(),
        meta: state.meta.wgpu_buffer().clone(),
        alpha: state.alpha.wgpu_buffer().clone(),
        beta: state.beta.wgpu_buffer().clone(),
        likelihood: state.likelihood.wgpu_buffer().clone(),
        divisors: state.divisors.wgpu_buffer().clone(),
    }
}

/// Compiled CTC steps using the native Metal-selected WGPU kernels.
#[derive(Debug)]
pub struct CtcKernel {
    inner: wgpu_backend::CtcKernel,
}

impl CtcKernel {
    /// Compile the steps for a Metal device.
    ///
    /// # Errors
    ///
    /// Returns the WGPU backend's kernel compilation failure.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        Ok(Self {
            inner: wgpu_backend::CtcKernel::new(device.wgpu_device())?,
        })
    }

    /// Run the alpha recurrence at frame `t` over Metal buffers.
    pub fn alpha_step(
        &self,
        device: &MetalDevice,
        t: u32,
        log_probs: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        let viewed = wgpu_state(state);
        self.inner
            .alpha_step(device.wgpu_device(), t, log_probs.wgpu_buffer(), &viewed)
    }

    /// Run the beta recurrence at frame `t` over Metal buffers.
    pub fn beta_step(
        &self,
        device: &MetalDevice,
        t: u32,
        log_probs: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        let viewed = wgpu_state(state);
        self.inner
            .beta_step(device.wgpu_device(), t, log_probs.wgpu_buffer(), &viewed)
    }

    /// Merge each sample's terminal alpha pair into its likelihood.
    pub fn loss_finish(
        &self,
        device: &MetalDevice,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        let viewed = wgpu_state(state);
        self.inner.loss_finish(device.wgpu_device(), &viewed)
    }

    /// Accumulate the seeded posterior into `grad` from retained `state`.
    pub fn posterior(
        &self,
        device: &MetalDevice,
        upstream: f32,
        grad: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        let viewed = wgpu_state(state);
        self.inner
            .posterior(device.wgpu_device(), upstream, grad.wgpu_buffer(), &viewed)
    }
}

/// Provider-owned implementation of [`CtcStepOps`] and [`CtcOps`] for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalCtcOps;

impl CtcStepOps<MetalDevice> for MetalCtcOps {
    type CtcStep = CtcKernel;

    fn prepare_ctc_step(&self, device: &MetalDevice) -> Result<Self::CtcStep> {
        CtcKernel::new(device)
    }

    fn ctc_alpha_step_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        kernel.alpha_step(device, t, log_probs, state)
    }

    fn ctc_beta_step_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        kernel.beta_step(device, t, log_probs, state)
    }

    fn ctc_loss_finish_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::CtcStep,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        kernel.loss_finish(device, state)
    }

    fn ctc_posterior_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::CtcStep,
        upstream: f32,
        grad: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}

impl CtcOps<MetalDevice> for MetalCtcOps {
    type Ctc = CtcKernel;

    fn prepare_ctc(&self, device: &MetalDevice) -> Result<Self::Ctc> {
        CtcKernel::new(device)
    }

    fn ctc_forward_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::Ctc,
        log_probs: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<f32> {
        hephaestus_core::drive_ctc_forward(self, device, kernel, log_probs, state)
    }

    fn ctc_backward_into(
        &self,
        device: &MetalDevice,
        kernel: &Self::Ctc,
        upstream: f32,
        grad: &MetalBuffer<f32>,
        state: &CtcStateBuffers<MetalDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}
