//! CTC operation seams and the shared driver.

use super::problem::invalid;
use super::state::CtcStateBuffers;
use crate::{ComputeDevice, Result};

/// One frame and time step of the device recurrences, plus the likelihood
/// finish and the posterior sweep.
///
/// This is the seam GPU backends bind. The frame walk itself lives in
/// [`drive_ctc_forward`], so every backend runs the same walk over its own
/// steps.
pub trait CtcStepOps<D: ComputeDevice> {
    /// Compiled step kernels, reusable across problems on one device.
    type CtcStep;

    /// Compile the step kernels for a device.
    ///
    /// # Errors
    ///
    /// Returns the backend's kernel compilation or layout failure.
    fn prepare_ctc_step(&self, device: &D) -> Result<Self::CtcStep>;

    /// Run the alpha recurrence at frame `t` over all samples and states.
    ///
    /// Frame 0 initializes the reachable prefixes; later frames merge
    /// predecessors and add the emission, exactly the provider recurrence.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn ctc_alpha_step_into(
        &self,
        device: &D,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &D::Buffer<f32>,
        state: &CtcStateBuffers<D>,
    ) -> Result<()>;

    /// Run the beta recurrence at frame `t` over all samples and states.
    ///
    /// The terminal frame initializes the accepted suffixes to zero; earlier
    /// frames merge successors with the next emission, exactly the provider
    /// recurrence, which excludes the current emission.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn ctc_beta_step_into(
        &self,
        device: &D,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &D::Buffer<f32>,
        state: &CtcStateBuffers<D>,
    ) -> Result<()>;

    /// Merge each sample's terminal alpha pair into its likelihood.
    ///
    /// Empty-frame samples take likelihood zero for an empty target and
    /// unreachable otherwise, exactly the provider's edge table. No lane
    /// reads alpha when its sample has zero frames, so backends that refuse
    /// to bind an empty alpha buffer may soundly substitute a stand-in for
    /// degenerate problems.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn ctc_loss_finish_into(
        &self,
        device: &D,
        kernel: &Self::CtcStep,
        state: &CtcStateBuffers<D>,
    ) -> Result<()>;

    /// Accumulate the seeded posterior into `grad` over all active lanes.
    ///
    /// Each `(sample, frame, class)` lane sums its states' posterior
    /// occupancy and adds `-upstream * posterior / (target * batch)`.
    /// Callers must not run this when forward reported a non-finite loss.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn ctc_posterior_into(
        &self,
        device: &D,
        kernel: &Self::CtcStep,
        upstream: f32,
        grad: &D::Buffer<f32>,
        state: &CtcStateBuffers<D>,
    ) -> Result<()>;
}

/// Device-neutral CTC forward and backward.
///
/// The uniform seam every backend binds, host included. GPU coarse methods
/// are one call into the shared [`drive_ctc_forward`] driver plus the
/// posterior step; the host runs the provider directly.
pub trait CtcOps<D: ComputeDevice> {
    /// Compiled CTC kernels, reusable across problems on one device.
    type Ctc;

    /// Compile the CTC kernels for a device.
    ///
    /// # Errors
    ///
    /// Returns the backend's kernel compilation or layout failure.
    fn prepare_ctc(&self, device: &D) -> Result<Self::Ctc>;

    /// Run forward over `log_probs`, fill `state`, and return the mean loss.
    ///
    /// `log_probs` is a contiguous `[frames, batch, classes]` grid of
    /// nonpositive lanes. An impossible alignment yields positive infinity,
    /// including a nonempty target with zero valid frames.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure, or a non-finite accumulation
    /// the provider would equally reject.
    fn ctc_forward_into(
        &self,
        device: &D,
        kernel: &Self::Ctc,
        log_probs: &D::Buffer<f32>,
        state: &CtcStateBuffers<D>,
    ) -> Result<f32>;

    /// Accumulate the seeded derivative into `grad` from retained `state`.
    ///
    /// Must not run when forward reported a non-finite loss.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn ctc_backward_into(
        &self,
        device: &D,
        kernel: &Self::Ctc,
        upstream: f32,
        grad: &D::Buffer<f32>,
        state: &CtcStateBuffers<D>,
    ) -> Result<()>;
}

/// Walk the shared forward pass over any step implementation.
///
/// Alpha sweeps frames upward, beta downward, the likelihood finish merges
/// the terminal pairs, and the host sums the per-sample contributions in
/// sample order — the provider's accumulation with its overflow checks.
///
/// # Errors
///
/// Returns the step dispatch failure, the likelihood readback failure, or a
/// non-finite accumulation the provider would equally reject.
pub fn drive_ctc_forward<D, S>(
    ops: &S,
    device: &D,
    kernel: &S::CtcStep,
    log_probs: &D::Buffer<f32>,
    state: &CtcStateBuffers<D>,
) -> Result<f32>
where
    D: ComputeDevice,
    S: CtcStepOps<D>,
{
    let frames = state.problem.frames();
    for t in 0..frames {
        let t = u32::try_from(t).map_err(|_| invalid(format!("ctc frame {t} does not fit u32")))?;
        ops.ctc_alpha_step_into(device, kernel, t, log_probs, state)?;
    }
    for t in (0..frames).rev() {
        let t = u32::try_from(t).map_err(|_| invalid(format!("ctc frame {t} does not fit u32")))?;
        ops.ctc_beta_step_into(device, kernel, t, log_probs, state)?;
    }
    ops.ctc_loss_finish_into(device, kernel, state)?;
    let batch = state.problem.batch();
    let mut likelihood = vec![0.0_f32; batch * 2];
    device.download(&state.likelihood, &mut likelihood)?;
    let batch_divisor = state.problem.batch_divisor();
    let mut loss = 0.0_f32;
    for sample in 0..batch {
        let high = likelihood[sample * 2];
        let value = high + likelihood[sample * 2 + 1];
        let divisor = state.problem.divisors()[sample];
        let contribution = (-value / divisor) / batch_divisor;
        // Reachability is the high lane exactly, as the provider checks it:
        // only a negative-infinity high lane takes the unreachable path.
        let reachable = high != f32::NEG_INFINITY;
        if reachable && !contribution.is_finite() {
            return Err(invalid(format!(
                "ctc sample {sample} contribution is not finite"
            )));
        }
        let previous = loss;
        loss += contribution;
        if loss.is_nan() || (previous.is_finite() && contribution.is_finite() && !loss.is_finite())
        {
            return Err(invalid(format!("ctc sample {sample} loss is not finite")));
        }
    }
    Ok(loss)
}
