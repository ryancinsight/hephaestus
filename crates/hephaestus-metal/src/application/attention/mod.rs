//! Zero-copy Metal attention delegation through the WGPU Metal backend.

mod operands;
mod prepared;
mod seam;

pub use seam::MetalAttentionOps;
