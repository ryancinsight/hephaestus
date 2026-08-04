//! Zero-copy cross-entropy delegation through the WGPU Metal backend.

mod operands;
mod prepared;
mod seam;

pub use seam::MetalCrossEntropyOps;
