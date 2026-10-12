use super::invalid;
use crate::Result;

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
    pub(super) batch: usize,
    pub(super) classes: usize,
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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration) for any problem the
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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration) when the grid shape
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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration) for a grid-shape
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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration) for a grid-shape
    /// overflow or any length mismatch.
    pub fn validate_storage(&self, log_probs_len: usize, grad_len: usize) -> Result<()> {
        self.validate_grid_len("log-probability grid", log_probs_len)?;
        self.validate_grid_len("gradient grid", grad_len)?;
        Ok(())
    }

    /// Check state-buffer lengths against the problem's derived geometry.
    ///
    /// State normally comes from [`CtcStateBuffers::allocate`](super::CtcStateBuffers::allocate), which builds
    /// it right; this guards hand-built state before any dispatch reads it.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration) for any length
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
