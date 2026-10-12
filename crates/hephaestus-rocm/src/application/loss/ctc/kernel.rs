use hephaestus_core::{
    CtcBackwardParams, CtcProblem, CtcStateBuffers, CtcStepParams, DeviceBuffer, DispatchGrid,
    MultiStorageKernel, Result,
};

use super::source::{CTC_KERNEL, FINISH_BLOCK, POSTERIOR_BLOCK, STEP_BLOCK};
use crate::RocmDevice;
use crate::application::storage_kernel::{RocmMultiStorageKernel, RocmStorageBinding};
use crate::infrastructure::RocmBuffer;

/// Compiled HIP CTC step kernels.
///
/// Built once per device and reused across problems; the frame and the
/// seed ride in each launch's parameter block.
#[derive(Debug)]
pub struct CtcKernel {
    alpha: RocmMultiStorageKernel,
    beta: RocmMultiStorageKernel,
    loss_finish: RocmMultiStorageKernel,
    posterior: RocmMultiStorageKernel,
}

impl CtcKernel {
    /// Compile the four CTC entries for a ROCm device.
    ///
    /// # Errors
    ///
    /// Returns the runtime compiler or module-load failure.
    pub fn new(_device: &RocmDevice) -> Result<Self> {
        Ok(Self {
            alpha: RocmMultiStorageKernel::new(
                "hephaestus-ctc-alpha",
                CTC_KERNEL,
                "ctc_alpha",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            beta: RocmMultiStorageKernel::new(
                "hephaestus-ctc-beta",
                CTC_KERNEL,
                "ctc_beta",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            loss_finish: RocmMultiStorageKernel::new(
                "hephaestus-ctc-loss-finish",
                CTC_KERNEL,
                "ctc_loss_finish",
                &[0, 1, 2],
                FINISH_BLOCK,
                0,
            )?,
            posterior: RocmMultiStorageKernel::new(
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
        state: &CtcStateBuffers<RocmDevice>,
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
        device: &RocmDevice,
        t: u32,
        log_probs: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        MultiStorageKernel::<RocmDevice, CtcStepParams, [RocmStorageBinding<'_>; 4]>::dispatch(
            &self.alpha,
            device,
            [
                RocmStorageBinding::new(0, log_probs),
                RocmStorageBinding::new(1, &state.labels),
                RocmStorageBinding::new(2, &state.meta),
                RocmStorageBinding::new(3, &state.alpha),
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
        device: &RocmDevice,
        t: u32,
        log_probs: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        MultiStorageKernel::<RocmDevice, CtcStepParams, [RocmStorageBinding<'_>; 4]>::dispatch(
            &self.beta,
            device,
            [
                RocmStorageBinding::new(0, log_probs),
                RocmStorageBinding::new(1, &state.labels),
                RocmStorageBinding::new(2, &state.meta),
                RocmStorageBinding::new(3, &state.beta),
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
        device: &RocmDevice,
        state: &CtcStateBuffers<RocmDevice>,
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
        MultiStorageKernel::<RocmDevice, CtcStepParams, [RocmStorageBinding<'_>; 3]>::dispatch(
            &self.loss_finish,
            device,
            [
                RocmStorageBinding::new(0, &state.meta),
                RocmStorageBinding::new(1, &state.alpha),
                RocmStorageBinding::new(2, &state.likelihood),
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
        device: &RocmDevice,
        upstream: f32,
        grad: &RocmBuffer<f32>,
        state: &CtcStateBuffers<RocmDevice>,
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
        MultiStorageKernel::<RocmDevice, CtcBackwardParams, [RocmStorageBinding<'_>; 7]>::dispatch(
            &self.posterior,
            device,
            [
                RocmStorageBinding::new(0, &state.labels),
                RocmStorageBinding::new(1, &state.meta),
                RocmStorageBinding::new(2, &state.alpha),
                RocmStorageBinding::new(3, &state.beta),
                RocmStorageBinding::new(4, &state.likelihood),
                RocmStorageBinding::new(5, &state.divisors),
                RocmStorageBinding::new(6, grad),
            ],
            &params,
            grid,
        )
    }
}
