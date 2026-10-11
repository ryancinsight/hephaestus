//! Runtime-rank expression fusion implemented by the CUDA provider.
//!
//! Kernel source, layout metadata, and the scalar vocabulary are owned by
//! `hephaestus-core` and monomorphized here over [`CudaC`](hephaestus_core::CudaC);
//! this module owns validation, pipeline caching, and launch only.

mod dispatch;

pub use dispatch::CudaFusionOps;
