//! Runtime-rank expression fusion implemented by the WGPU provider.
//!
//! The scalar vocabulary is owned by `hephaestus-core` and bound here as
//! [`FusionScalar`](hephaestus_core::FusionScalar)`<Wgsl>`; the WGSL source,
//! layout metadata, and dispatch below stay provider-local because their
//! binding model and address widths are a different ABI from the C dialects.

mod dispatch;
mod source;

pub use dispatch::WgpuFusionOps;
