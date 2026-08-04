//! WGPU provider implementation for mean cross-entropy.

mod metadata;
mod prepared;
mod resources;
mod seam;
mod shader;

pub use prepared::{PreparedCrossEntropyBackward, PreparedCrossEntropyForward};
pub use seam::WgpuCrossEntropyOps;

#[cfg(test)]
mod tests;
