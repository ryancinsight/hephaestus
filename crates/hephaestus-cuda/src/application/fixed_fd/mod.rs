//! CUDA fixed-scheme three-dimensional first-derivative dispatch.
//!
//! The CUDA counterpart of the WGPU sweep, computing the same five schemes
//! from the same parameter block. One entry point serves every scheme behind
//! a uniform branch, and the per-lane arithmetic keeps the provider's exact
//! operation order. `f32` only, matching
//! [`FixedFd3DParams`](hephaestus_core::FixedFd3DParams)'s lane layout.

mod kernel;
mod ops;
mod source;

pub use hephaestus_core::{FixedFd3DParams, FixedFd3DScheme};
pub use kernel::FixedFd3DKernel;
pub use ops::CudaFixedFd3DOps;
