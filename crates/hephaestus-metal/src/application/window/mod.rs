//! Zero-copy pooling and sliding-window delegation to Metal-selected WGPU.

mod operands;
mod prepared;
mod seam;

pub use seam::MetalPoolingOps;
pub use seam::MetalSlidingWindowOps;
