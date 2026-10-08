//! Zero-copy cross-entropy delegation through the WGPU Metal backend.

mod ctc;
mod operands;
mod prepared;
mod seam;

pub use ctc::{CtcKernel, MetalCtcOps};
pub use seam::MetalCrossEntropyOps;
