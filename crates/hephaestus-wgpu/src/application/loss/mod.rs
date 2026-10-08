//! WGPU provider implementation for mean cross-entropy.

mod ctc;
mod metadata;
mod prepared;
mod resources;
mod seam;
mod shader;

pub use ctc::{CtcKernel, WgpuCtcOps};
pub use prepared::{PreparedCrossEntropyBackward, PreparedCrossEntropyForward};
pub use seam::WgpuCrossEntropyOps;

#[cfg(test)]
mod tests;
