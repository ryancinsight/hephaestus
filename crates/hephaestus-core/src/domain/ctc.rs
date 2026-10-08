//! Backend-neutral connectionist temporal classification.
//!
//! # What this is for
//!
//! The Graves forward-backward loss over `[frames, batch, classes]`
//! log-probabilities. Leto owns it on the CPU as `leto_ops::CtcState`; this
//! is the device side of the same contract, so one loss reaches either
//! backend.
//!
//! # Why steps, and why two traits
//!
//! The frame recurrence is sequential — frame `t` reads frame `t − 1` — so no
//! single dispatch can span time. The seam therefore exposes one step per
//! frame and time step (`alpha`, `beta`), a per-sample likelihood finish,
//! and the embarrassingly parallel posterior, and a single shared driver in
//! this module walks them. A backend that ran its own loop would duplicate
//! the walk five times over; a host that faked steps it never takes would be
//! a mock wearing a trait impl. So [`CtcStepOps`] is the step seam GPUs bind,
//! [`CtcOps`] is the coarse seam every backend — host included — binds, and
//! GPU coarse methods are one call into the shared driver.
//!
//! # Log-space arithmetic
//!
//! Messages are `Weight` pairs `(high, low)` in the provider's compensated
//! form: TwoSum addition, log-sum-exp merge through `ln(1 + exp(d))`, and
//! `-infinity` for unreachable paths. Kernels reproduce the provider's
//! operation order exactly, including the large/small selection tie-break,
//! so CPU and device agree up to elementary-function rounding. Buffers hold
//! two `f32` lanes per message; the second lane of each pair is the residual.
//!
//! # What the driver keeps on the device
//!
//! [`CtcStateBuffers`] owns the forward recurrences between the passes —
//! alpha and beta, the expanded labels, per-sample metadata, likelihoods,
//! and divisors — allocated once per problem and retained across backward,
//! exactly the provider state's device image. The scalar loss is summed on
//! the host from the per-sample likelihoods in sample order, mirroring the
//! provider's accumulation and its overflow checks.
//!
//! # Preconditions the device does not check
//!
//! Active log-probabilities must be nonpositive finite-or-`-infinity`
//! lanes — log-softmax outputs satisfy this — the compensated intermediates
//! must stay finite, which holds for every representable problem, and
//! backward must not run when forward reported a non-finite loss: the
//! provider errors on impossible alignments, while a kernel can only write
//! the zeros the posterior takes there. Callers check the returned loss
//! first; the conformance suite pins the forward infinities and skips those
//! backward passes. The upstream seed, by contrast, is checked on the host
//! where the check is free.

use eunomia::{Pod, Zeroable};

use crate::{ComputeDevice, HephaestusError, Result};

/// Per-sample derived geometry: valid frames, expanded states, and the
/// message and label offsets into the packed buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtcSampleMeta {
    /// Valid frames for the sample (the input-length prefix).
    pub frames: usize,
    /// Expanded states `2 * target_length + 1`.
    pub states: usize,
    /// Message-cell offset of frame 0 into alpha/beta.
    pub offset: usize,
    /// Label offset of state 0 into the expanded labels.
    pub labels_offset: usize,
}

/// A validated CTC problem with its device upload vectors precomputed.
///
/// Construction mirrors the provider's rejection set: empty batch or class
/// counts, length-count mismatches, out-of-range blanks and labels, input
/// lengths past the frame count, and target-count mismatches all fail here,
/// before any device work. Offsets additionally must fit `u32`, since lanes
/// index with 32-bit arithmetic.
#[derive(Debug, Clone)]
pub struct CtcProblem {
    frames: usize,
    batch: usize,
    classes: usize,
    samples: Vec<CtcSampleMeta>,
    /// Expanded labels with interleaved blanks, one `states` run per sample.
    labels: Vec<u32>,
    /// Four words per sample: frames, states, message offset, label offset.
    meta: Vec<u32>,
    /// `max(target_length, 1)` per sample, the forward's target divisor.
    divisors: Vec<f32>,
    total_cells: usize,
    max_states: usize,
    max_frames: usize,
}

impl CtcProblem {
    /// Validate a problem and precompute its upload vectors.
    ///
    /// `targets` concatenates every sample's labels (blank excluded);
    /// `input_lengths` and `target_lengths` have one entry per batch member.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] for any problem the
    /// provider rejects, or when message offsets do not fit `u32`.
    pub fn new(
        frames: usize,
        batch: usize,
        classes: usize,
        blank: usize,
        input_lengths: &[usize],
        target_lengths: &[usize],
        targets: &[usize],
    ) -> Result<Self> {
        if batch == 0 || classes == 0 {
            return Err(invalid(format!(
                "ctc problem needs a nonempty batch and class count, got batch={batch} classes={classes}"
            )));
        }
        for (name, extent) in [("frames", frames), ("batch", batch), ("classes", classes)] {
            u32::try_from(extent)
                .map_err(|_| invalid(format!("ctc {name} extent {extent} does not fit u32")))?;
        }
        if input_lengths.len() != batch || target_lengths.len() != batch {
            return Err(invalid(format!(
                "ctc length counts must match batch={batch}, got inputs={} targets={}",
                input_lengths.len(),
                target_lengths.len(),
            )));
        }
        if blank >= classes {
            return Err(invalid(format!(
                "ctc blank {blank} is outside {classes} classes"
            )));
        }
        let target_count: usize = target_lengths.iter().sum();
        if target_count != targets.len() {
            return Err(invalid(format!(
                "ctc targets hold {} labels for {target_count} claimed",
                targets.len()
            )));
        }
        for (index, &label) in targets.iter().enumerate() {
            if label == blank || label >= classes {
                return Err(invalid(format!(
                    "ctc target {index} holds label {label} outside the non-blank classes"
                )));
            }
        }
        let blank_word = u32::try_from(blank)
            .map_err(|_| invalid(format!("ctc blank {blank} does not fit u32")))?;
        let mut samples = Vec::with_capacity(batch);
        let mut labels = Vec::new();
        let mut meta = Vec::with_capacity(batch * 4);
        let mut divisors = Vec::with_capacity(batch);
        let mut offset = 0_usize;
        let mut target_offset = 0_usize;
        let mut max_states = 0_usize;
        let mut max_frames = 0_usize;
        for (sample, (&length, &target_length)) in
            input_lengths.iter().zip(target_lengths).enumerate()
        {
            if length > frames {
                return Err(invalid(format!(
                    "ctc sample {sample} claims {length} frames of {frames}"
                )));
            }
            let states = target_length
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| invalid(format!("ctc sample {sample} state count overflows")))?;
            let cells = length
                .checked_mul(states)
                .ok_or_else(|| invalid(format!("ctc sample {sample} message grid overflows")))?;
            let labels_offset = labels.len();
            labels.push(blank_word);
            for &label in &targets[target_offset..target_offset + target_length] {
                let word = u32::try_from(label).map_err(|_| {
                    invalid(format!(
                        "ctc sample {sample} label {label} does not fit u32"
                    ))
                })?;
                labels.extend([word, blank_word]);
            }
            target_offset += target_length;
            for word in [length, states, offset, labels_offset] {
                let word = u32::try_from(word).map_err(|_| {
                    invalid(format!(
                        "ctc sample {sample} metadata {word} does not fit u32"
                    ))
                })?;
                meta.push(word);
            }
            let divisor = target_length.max(1) as f32;
            if !divisor.is_finite() {
                return Err(invalid(format!(
                    "ctc sample {sample} target divisor is not finite"
                )));
            }
            divisors.push(divisor);
            samples.push(CtcSampleMeta {
                frames: length,
                states,
                offset,
                labels_offset,
            });
            offset = offset
                .checked_add(cells)
                .ok_or_else(|| invalid("ctc message grid overflows usize".to_owned()))?;
            max_states = max_states.max(states);
            max_frames = max_frames.max(length);
        }
        u32::try_from(offset)
            .map_err(|_| invalid(format!("ctc message cells {offset} do not fit u32")))?;
        u32::try_from(labels.len())
            .map_err(|_| invalid(format!("ctc labels {} do not fit u32", labels.len())))?;
        Ok(Self {
            frames,
            batch,
            classes,
            samples,
            labels,
            meta,
            divisors,
            total_cells: offset,
            max_states,
            max_frames,
        })
    }

    /// Frame count of the log-probability grid.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Batch size.
    #[must_use]
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Class count.
    #[must_use]
    pub const fn classes(&self) -> usize {
        self.classes
    }

    /// Per-sample derived geometry.
    #[must_use]
    pub fn samples(&self) -> &[CtcSampleMeta] {
        &self.samples
    }

    /// Expanded labels with interleaved blanks.
    #[must_use]
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// Four metadata words per sample.
    #[must_use]
    pub fn meta(&self) -> &[u32] {
        &self.meta
    }

    /// Per-sample target divisors.
    #[must_use]
    pub fn divisors(&self) -> &[f32] {
        &self.divisors
    }

    /// Total alpha/beta message cells across samples.
    #[must_use]
    pub const fn total_cells(&self) -> usize {
        self.total_cells
    }

    /// Widest expanded state count, for grid sizing.
    #[must_use]
    pub const fn max_states(&self) -> usize {
        self.max_states
    }

    /// Longest valid frame prefix, for grid sizing.
    #[must_use]
    pub const fn max_frames(&self) -> usize {
        self.max_frames
    }

    /// Batch divisor as the device arithmetic sees it.
    #[must_use]
    pub fn batch_divisor(&self) -> f32 {
        self.batch as f32
    }

    /// Expected element count of the contiguous log-probability and gradient
    /// grids.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when the grid shape
    /// overflows.
    pub fn grid_len(&self) -> Result<usize> {
        self.frames
            .checked_mul(self.batch)
            .and_then(|n| n.checked_mul(self.classes))
            .ok_or_else(|| invalid("ctc grid shape overflows usize".to_owned()))
    }

    /// Check one grid buffer against the problem's grid shape.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] for a grid-shape
    /// overflow or a length mismatch.
    pub fn validate_grid_len(&self, what: &str, actual: usize) -> Result<()> {
        let expected = self.grid_len()?;
        if actual != expected {
            return Err(invalid(format!(
                "ctc {what} holds {actual} lanes for {expected} expected"
            )));
        }
        Ok(())
    }

    /// Check device-buffer lengths against the problem's grid shape.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] for a grid-shape
    /// overflow or any length mismatch.
    pub fn validate_storage(&self, log_probs_len: usize, grad_len: usize) -> Result<()> {
        self.validate_grid_len("log-probability grid", log_probs_len)?;
        self.validate_grid_len("gradient grid", grad_len)?;
        Ok(())
    }

    /// Check state-buffer lengths against the problem's derived geometry.
    ///
    /// State normally comes from [`CtcStateBuffers::allocate`], which builds
    /// it right; this guards hand-built state before any dispatch reads it.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] for any length
    /// mismatch.
    pub fn validate_state_lengths(
        &self,
        labels: usize,
        meta: usize,
        alpha: usize,
        beta: usize,
        likelihood: usize,
        divisors: usize,
    ) -> Result<()> {
        let expected = [
            ("label", labels, self.labels.len()),
            ("metadata", meta, self.batch * 4),
            ("alpha", alpha, self.total_cells * 2),
            ("beta", beta, self.total_cells * 2),
            ("likelihood", likelihood, self.batch * 2),
            ("divisor", divisors, self.batch),
        ];
        for (what, actual, wanted) in expected {
            if actual != wanted {
                return Err(invalid(format!(
                    "ctc {what} buffer holds {actual} lanes for {wanted} expected"
                )));
            }
        }
        Ok(())
    }
}

fn invalid(message: String) -> HephaestusError {
    HephaestusError::InvalidConfiguration { message }
}

/// Uniform block shared by the alpha, beta, and likelihood steps.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CtcStepParams {
    /// `(frame, batch, classes, 0)`.
    pub dims: [u32; 4],
}

impl CtcStepParams {
    /// Build the uniform block for one step at frame `t`.
    #[must_use]
    pub fn for_frame(problem: &CtcProblem, t: u32) -> Self {
        Self {
            dims: [t, problem.batch as u32, problem.classes as u32, 0],
        }
    }
}

/// Uniform block for the posterior sweep.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CtcBackwardParams {
    /// `(0, batch, classes, 0)`: the frame slot is zero, and batch and
    /// classes sit where the step block keeps them, so every entry reads
    /// `dims.y` as the batch and `dims.z` as the classes.
    pub dims: [u32; 4],
    /// `(upstream, 0, 0, 0)`.
    pub scalars: [f32; 4],
}

impl CtcBackwardParams {
    /// Build the uniform block for one backward pass.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when the seed is not
    /// finite, which the provider equally rejects.
    pub fn new(problem: &CtcProblem, upstream: f32) -> Result<Self> {
        if !upstream.is_finite() {
            return Err(invalid("ctc upstream seed is not finite".to_owned()));
        }
        Ok(Self {
            dims: [0, problem.batch as u32, problem.classes as u32, 0],
            scalars: [upstream, 0.0, 0.0, 0.0],
        })
    }
}

/// The forward recurrences on the device, retained across backward.
///
/// This is the provider state's device image: alpha and beta as two `f32`
/// lanes per message cell, the expanded labels and per-sample metadata, the
/// per-sample likelihood pairs and target divisors. Allocated once per
/// problem; forward fills it, backward reads it.
pub struct CtcStateBuffers<D: ComputeDevice> {
    /// The validated host problem this state was allocated for.
    pub problem: CtcProblem,
    /// Expanded labels, `total_labels` words.
    pub labels: D::Buffer<u32>,
    /// Four metadata words per sample.
    pub meta: D::Buffer<u32>,
    /// Alpha messages, two `f32` lanes per cell.
    pub alpha: D::Buffer<f32>,
    /// Beta messages, two `f32` lanes per cell.
    pub beta: D::Buffer<f32>,
    /// Log-likelihood pairs, two lanes per sample.
    pub likelihood: D::Buffer<f32>,
    /// Target divisors, one per sample.
    pub divisors: D::Buffer<f32>,
}

impl<D: ComputeDevice> CtcStateBuffers<D> {
    /// Allocate state for a problem, uploading labels, metadata, and
    /// divisors and zeroing the recurrences the forward pass fills.
    ///
    /// # Errors
    ///
    /// Returns the backend's allocation or upload failure.
    pub fn allocate(device: &D, problem: CtcProblem) -> Result<Self> {
        let labels = device.upload(problem.labels())?;
        let meta = device.upload(problem.meta())?;
        let divisors = device.upload(problem.divisors())?;
        let lanes = problem
            .total_cells()
            .checked_mul(2)
            .ok_or_else(|| invalid("ctc message lanes overflow usize".to_owned()))?;
        let alpha = device.alloc_zeroed::<f32>(lanes)?;
        let beta = device.alloc_zeroed::<f32>(lanes)?;
        let likelihood = device.alloc_zeroed::<f32>(problem.batch() * 2)?;
        Ok(Self {
            problem,
            labels,
            meta,
            alpha,
            beta,
            likelihood,
            divisors,
        })
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> CtcProblem {
        CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1, 2]).expect("a valid problem")
    }

    #[test]
    fn upload_vectors_expand_labels_and_metadata() {
        let problem = problem();
        // Sample 0 interleaves blanks around [1, 2]; sample 1 is blank-only.
        assert_eq!(problem.labels(), &[0, 1, 0, 2, 0, 0]);
        assert_eq!(problem.meta(), &[4, 5, 0, 0, 3, 1, 20, 5]);
        assert_eq!(problem.divisors(), &[2.0, 1.0]);
        // Four frames of five states plus three frames of one state.
        assert_eq!(problem.total_cells(), 23);
        assert_eq!(problem.max_states(), 5);
        assert_eq!(problem.max_frames(), 4);
    }

    #[test]
    fn validation_mirrors_the_provider_rejections() {
        assert!(CtcProblem::new(4, 0, 6, 0, &[], &[], &[]).is_err());
        assert!(CtcProblem::new(4, 2, 0, 0, &[4, 3], &[2, 0], &[1, 2]).is_err());
        assert!(CtcProblem::new(4, 2, 6, 0, &[4], &[2, 0], &[1, 2]).is_err());
        assert!(CtcProblem::new(4, 2, 6, 6, &[4, 3], &[2, 0], &[1, 2]).is_err());
        assert!(CtcProblem::new(4, 2, 6, 0, &[5, 3], &[2, 0], &[1, 2]).is_err());
        assert!(CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1]).is_err());
        assert!(CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1, 0]).is_err());
        assert!(CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1, 6]).is_err());
        assert_eq!(problem().batch_divisor(), 2.0);
    }

    #[test]
    fn storage_validation_covers_both_grids() {
        let problem = problem();
        assert_eq!(problem.grid_len().expect("a grid length"), 4 * 2 * 6);
        assert!(problem.validate_storage(4 * 2 * 6, 4 * 2 * 6).is_ok());
        assert!(problem.validate_storage(4 * 2 * 6 - 1, 4 * 2 * 6).is_err());
        assert!(problem.validate_storage(4 * 2 * 6, 4 * 2 * 6 + 1).is_err());
    }

    #[test]
    fn backward_params_reject_a_non_finite_seed() {
        let problem = problem();
        assert!(CtcBackwardParams::new(&problem, 1.0).is_ok());
        assert!(CtcBackwardParams::new(&problem, f32::NAN).is_err());
        assert!(CtcBackwardParams::new(&problem, f32::INFINITY).is_err());
    }

    #[test]
    fn both_param_blocks_share_the_dims_convention() {
        let problem = problem();
        let step = CtcStepParams::for_frame(&problem, 3);
        let backward = CtcBackwardParams::new(&problem, 1.5).expect("a seed");
        // Every entry reads dims.y as the batch and dims.z as the classes;
        // the posterior's frame slot is zero.
        assert_eq!(step.dims, [3, 2, 6, 0]);
        assert_eq!(backward.dims, [0, 2, 6, 0]);
        assert_eq!(backward.scalars[0], 1.5);
    }
}
