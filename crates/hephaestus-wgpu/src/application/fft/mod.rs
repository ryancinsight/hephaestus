//! Provider-owned WGPU dense complex FFT planning and dispatch.

mod dispatch;
mod kernel;
mod pipelines;
mod plan;
mod seam;
mod stages;
mod strategy;

pub use seam::{WgpuFftOps, WgpuPreparedFft};
