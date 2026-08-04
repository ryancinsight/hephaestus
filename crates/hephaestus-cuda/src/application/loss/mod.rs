//! Prepared CUDA mean cross-entropy kernels.

mod kernel;
mod metadata;
mod prepared;
mod resources;
mod seam;

pub use seam::CudaCrossEntropyOps;
