use super::state::CtcStateBuffers;
use crate::{ComputeDevice, Result};

/// One frame and time step of the device recurrences, plus the likelihood
/// finish and the posterior sweep.
///
/// This is the seam GPU backends bind. The frame walk itself lives in
/// [`drive_ctc_forward`](super::drive_ctc_forward), so every backend runs the
/// same walk over its own steps.
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
/// are one call into the shared [`drive_ctc_forward`](super::drive_ctc_forward)
/// driver plus the posterior step; the host runs the provider directly.
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
