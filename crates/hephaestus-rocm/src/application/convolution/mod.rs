//! Native ROCm implementation of the provider-owned convolution seam.

mod kernel;
mod metadata;
mod prepared;
mod resources;
mod routing;
mod seam;

pub use seam::RocmConvolutionOps;
