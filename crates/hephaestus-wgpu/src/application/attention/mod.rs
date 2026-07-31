//! WGPU provider implementation for scaled dot-product attention.

mod metadata;
mod prepared;
mod resources;
mod seam;
mod shader;

pub use seam::WgpuAttentionOps;

#[cfg(test)]
mod tests;
