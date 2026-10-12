use super::invalid;
use super::problem::CtcProblem;
use crate::{ComputeDevice, Result};

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
