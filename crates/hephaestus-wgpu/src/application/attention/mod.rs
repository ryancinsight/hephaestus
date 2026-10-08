//! WGPU provider implementation for scaled dot-product attention.

mod metadata;
mod preflight;
mod prepared;
mod resources;
mod scalar;
mod seam;
mod shader;

pub(in crate::application::attention) use scalar::WgslAttentionScalar;
pub use seam::WgpuAttentionOps;

#[cfg(test)]
mod tests;
