use hephaestus_core::{
    CtcBackwardParams, CtcProblem, CtcStateBuffers, CtcStepParams, DeviceBuffer, DispatchGrid,
    MultiStorageKernel, Result,
};

use super::source::{CTC_KERNEL, FINISH_BLOCK, POSTERIOR_BLOCK, STEP_BLOCK};
use crate::CudaDevice;
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::infrastructure::buffer::CudaBuffer;

/// Compiled CUDA CTC step kernels.
///
/// Built once per device and reused across problems; the frame and the
/// seed ride in each launch's parameter block.
#[derive(Debug)]
pub struct CtcKernel {
    alpha: CudaMultiStorageKernel,
    beta: CudaMultiStorageKernel,
    loss_finish: CudaMultiStorageKernel,
    posterior: CudaMultiStorageKernel,
}

impl CtcKernel {
    /// Compile the four CTC entries for a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns the runtime compiler or module-load failure.
    pub fn new(_device: &CudaDevice) -> Result<Self> {
        Ok(Self {
            alpha: CudaMultiStorageKernel::new(
                "hephaestus-ctc-alpha",
                CTC_KERNEL,
                "ctc_alpha",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            beta: CudaMultiStorageKernel::new(
                "hephaestus-ctc-beta",
                CTC_KERNEL,
                "ctc_beta",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            loss_finish: CudaMultiStorageKernel::new(
                "hephaestus-ctc-loss-finish",
                CTC_KERNEL,
                "ctc_loss_finish",
                &[0, 1, 2],
                FINISH_BLOCK,
                0,
            )?,
            posterior: CudaMultiStorageKernel::new(
                "hephaestus-ctc-posterior",
                CTC_KERNEL,
                "ctc_posterior",
                &[0, 1, 2, 3, 4, 5, 6],
                POSTERIOR_BLOCK,
                0,
            )?,
        })
    }

    fn validate_step(
        problem: &CtcProblem,
        log_probs_len: usize,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        problem.validate_grid_len("log-probability grid", log_probs_len)?;
        problem.validate_state_lengths(
            state.labels.len(),
            state.meta.len(),
            state.alpha.len(),
            state.beta.len(),
            state.likelihood.len(),
            state.divisors.len(),
        )
    }

    fn step_grid(problem: &CtcProblem) -> Result<DispatchGrid> {
        DispatchGrid::covering_domain(
            [problem.batch(), problem.max_states(), 1],
            [
                STEP_BLOCK[0] as usize,
                STEP_BLOCK[1] as usize,
                STEP_BLOCK[2] as usize,
            ],
        )
    }

    /// Run the alpha recurrence at frame `t`, `log_probs` into `state`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, or the
    /// launch failure.
    pub fn alpha_step(
        &self,
        device: &CudaDevice,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        MultiStorageKernel::<CudaDevice, CtcStepParams, [CudaStorageBinding<'_>; 4]>::dispatch(
            &self.alpha,
            device,
            [
                CudaStorageBinding::new(0, log_probs),
                CudaStorageBinding::new(1, &state.labels),
                CudaStorageBinding::new(2, &state.meta),
                CudaStorageBinding::new(3, &state.alpha),
            ],
            &CtcStepParams::for_frame(&state.problem, t),
            Self::step_grid(&state.problem)?,
        )
    }

    /// Run the beta recurrence at frame `t`, `log_probs` into `state`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, or the
    /// launch failure.
    pub fn beta_step(
        &self,
        device: &CudaDevice,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        MultiStorageKernel::<CudaDevice, CtcStepParams, [CudaStorageBinding<'_>; 4]>::dispatch(
            &self.beta,
            device,
            [
                CudaStorageBinding::new(0, log_probs),
                CudaStorageBinding::new(1, &state.labels),
                CudaStorageBinding::new(2, &state.meta),
                CudaStorageBinding::new(3, &state.beta),
            ],
            &CtcStepParams::for_frame(&state.problem, t),
            Self::step_grid(&state.problem)?,
        )
    }

    /// Merge each sample's terminal alpha pair into its likelihood.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, or the
    /// launch failure.
    pub fn loss_finish(
        &self,
        device: &CudaDevice,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        state.problem.validate_state_lengths(
            state.labels.len(),
            state.meta.len(),
            state.alpha.len(),
            state.beta.len(),
            state.likelihood.len(),
            state.divisors.len(),
        )?;
        let params = CtcStepParams::for_frame(&state.problem, 0);
        let grid = DispatchGrid::covering_domain(
            [state.problem.batch(), 1, 1],
            [
                FINISH_BLOCK[0] as usize,
                FINISH_BLOCK[1] as usize,
                FINISH_BLOCK[2] as usize,
            ],
        )?;
        MultiStorageKernel::<CudaDevice, CtcStepParams, [CudaStorageBinding<'_>; 3]>::dispatch(
            &self.loss_finish,
            device,
            [
                CudaStorageBinding::new(0, &state.meta),
                CudaStorageBinding::new(1, &state.alpha),
                CudaStorageBinding::new(2, &state.likelihood),
            ],
            &params,
            grid,
        )
    }

    /// Accumulate the seeded posterior into `grad` from retained `state`.
    ///
    /// Must not run when forward reported a non-finite loss.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, a non-finite
    /// seed, or the launch failure.
    pub fn posterior(
        &self,
        device: &CudaDevice,
        upstream: f32,
        grad: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        state
            .problem
            .validate_grid_len("gradient grid", grad.len())?;
        state.problem.validate_state_lengths(
            state.labels.len(),
            state.meta.len(),
            state.alpha.len(),
            state.beta.len(),
            state.likelihood.len(),
            state.divisors.len(),
        )?;
        let params = CtcBackwardParams::new(&state.problem, upstream)?;
        let grid = DispatchGrid::covering_domain(
            [
                state.problem.batch(),
                state.problem.max_frames(),
                state.problem.classes(),
            ],
            [
                POSTERIOR_BLOCK[0] as usize,
                POSTERIOR_BLOCK[1] as usize,
                POSTERIOR_BLOCK[2] as usize,
            ],
        )?;
        MultiStorageKernel::<CudaDevice, CtcBackwardParams, [CudaStorageBinding<'_>; 7]>::dispatch(
            &self.posterior,
            device,
            [
                CudaStorageBinding::new(0, &state.labels),
                CudaStorageBinding::new(1, &state.meta),
                CudaStorageBinding::new(2, &state.alpha),
                CudaStorageBinding::new(3, &state.beta),
                CudaStorageBinding::new(4, &state.likelihood),
                CudaStorageBinding::new(5, &state.divisors),
                CudaStorageBinding::new(6, grad),
            ],
            &params,
            grid,
        )
    }
}
