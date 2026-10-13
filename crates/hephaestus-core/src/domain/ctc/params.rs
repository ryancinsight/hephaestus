//! CTC kernel parameter blocks.

use super::problem::{CtcProblem, invalid};
use crate::Result;
use eunomia::{Pod, Zeroable};

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
            dims: [
                t,
                u32::try_from(problem.batch)
                    .expect("CTC batch is validated to u32 at construction"),
                u32::try_from(problem.classes)
                    .expect("CTC classes are validated to u32 at construction"),
                0,
            ],
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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration) when the seed is not
    /// finite, which the provider equally rejects.
    pub fn new(problem: &CtcProblem, upstream: f32) -> Result<Self> {
        if !upstream.is_finite() {
            return Err(invalid("ctc upstream seed is not finite".to_owned()));
        }
        Ok(Self {
            dims: [
                0,
                u32::try_from(problem.batch)
                    .expect("CTC batch is validated to u32 at construction"),
                u32::try_from(problem.classes)
                    .expect("CTC classes are validated to u32 at construction"),
                0,
            ],
            scalars: [upstream, 0.0, 0.0, 0.0],
        })
    }
}
