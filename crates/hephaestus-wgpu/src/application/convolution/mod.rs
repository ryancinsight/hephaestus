//! WGPU provider implementation for regular and transposed convolution.

mod metadata;
mod prepared;
mod resources;
mod routing;
mod seam;
mod shader;

pub use seam::WgpuConvolutionOps;

#[cfg(test)]
mod tests;
