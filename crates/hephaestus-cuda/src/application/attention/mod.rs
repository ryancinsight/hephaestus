//! Native CUDA scaled dot-product attention.

mod kernel;
mod metadata;
mod prepared;
mod resources;
mod seam;

pub use seam::CudaAttentionOps;
