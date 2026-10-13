//! Shared CTC step kernels for flat-argument backends.
//!
//! One monomorphized implementation serves CUDA C and HIP C: the alpha and
//! beta frame steps, the likelihood finish, and the posterior sweep,
//! dispatched through the [`MultiStorageKernel`](crate::MultiStorageKernel)
//! seam with the shared [`source`] text. Backends instantiate
//! `CtcKernel` with their own kernel type and bind the step traits to it.

mod source;

use source::{CTC_KERNEL, FINISH_BLOCK, POSTERIOR_BLOCK, STEP_BLOCK};

use super::params::{CtcBackwardParams, CtcStepParams};
use super::problem::CtcProblem;
use super::state::CtcStateBuffers;
use crate::{
    DeviceBuffer, DispatchGrid, FlatMultiStorageKernel, MultiStorageDevice, MultiStorageKernel,
    Result,
};

/// Compiled C-dialect CTC step kernels.
///
/// Built once per device and reused across problems; the frame and the
/// seed ride in each launch's parameter block.
#[derive(Debug)]
pub struct CtcKernel<K> {
    alpha: K,
    beta: K,
    loss_finish: K,
    posterior: K,
}

impl<K> CtcKernel<K> {
    /// Compile the four CTC entries for a flat-argument device.
    ///
    /// # Errors
    ///
    /// Returns the runtime compiler or module-load failure.
    pub fn new<D: MultiStorageDevice>(_device: &D) -> Result<Self>
    where
        K: FlatMultiStorageKernel,
    {
        Ok(Self {
            alpha: K::new(
                "hephaestus-ctc-alpha",
                CTC_KERNEL,
                "ctc_alpha",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            beta: K::new(
                "hephaestus-ctc-beta",
                CTC_KERNEL,
                "ctc_beta",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            loss_finish: K::new(
                "hephaestus-ctc-loss-finish",
                CTC_KERNEL,
                "ctc_loss_finish",
                &[0, 1, 2],
                FINISH_BLOCK,
                0,
            )?,
            posterior: K::new(
                "hephaestus-ctc-posterior",
                CTC_KERNEL,
                "ctc_posterior",
                &[0, 1, 2, 3, 4, 5, 6],
                POSTERIOR_BLOCK,
                0,
            )?,
        })
    }

    fn validate_step<D: crate::ComputeDevice>(
        problem: &CtcProblem,
        log_probs_len: usize,
        state: &CtcStateBuffers<D>,
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
    pub fn alpha_step<'s, D>(
        &self,
        device: &D,
        t: u32,
        log_probs: &'s D::Buffer<f32>,
        state: &'s CtcStateBuffers<D>,
    ) -> Result<()>
    where
        D: MultiStorageDevice + 's,
        K: MultiStorageKernel<D, CtcStepParams, [D::StorageBinding<'s>; 4]>,
    {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        self.alpha.dispatch(
            device,
            [
                D::storage_binding(0, log_probs),
                D::storage_binding(1, &state.labels),
                D::storage_binding(2, &state.meta),
                D::storage_binding(3, &state.alpha),
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
    pub fn beta_step<'s, D>(
        &self,
        device: &D,
        t: u32,
        log_probs: &'s D::Buffer<f32>,
        state: &'s CtcStateBuffers<D>,
    ) -> Result<()>
    where
        D: MultiStorageDevice + 's,
        K: MultiStorageKernel<D, CtcStepParams, [D::StorageBinding<'s>; 4]>,
    {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        self.beta.dispatch(
            device,
            [
                D::storage_binding(0, log_probs),
                D::storage_binding(1, &state.labels),
                D::storage_binding(2, &state.meta),
                D::storage_binding(3, &state.beta),
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
    pub fn loss_finish<'s, D>(&self, device: &D, state: &'s CtcStateBuffers<D>) -> Result<()>
    where
        D: MultiStorageDevice + 's,
        K: MultiStorageKernel<D, CtcStepParams, [D::StorageBinding<'s>; 3]>,
    {
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
        self.loss_finish.dispatch(
            device,
            [
                D::storage_binding(0, &state.meta),
                D::storage_binding(1, &state.alpha),
                D::storage_binding(2, &state.likelihood),
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
    pub fn posterior<'s, D>(
        &self,
        device: &D,
        upstream: f32,
        grad: &'s D::Buffer<f32>,
        state: &'s CtcStateBuffers<D>,
    ) -> Result<()>
    where
        D: MultiStorageDevice + 's,
        K: MultiStorageKernel<D, CtcBackwardParams, [D::StorageBinding<'s>; 7]>,
    {
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
        self.posterior.dispatch(
            device,
            [
                D::storage_binding(0, &state.labels),
                D::storage_binding(1, &state.meta),
                D::storage_binding(2, &state.alpha),
                D::storage_binding(3, &state.beta),
                D::storage_binding(4, &state.likelihood),
                D::storage_binding(5, &state.divisors),
                D::storage_binding(6, grad),
            ],
            &params,
            grid,
        )
    }
}
