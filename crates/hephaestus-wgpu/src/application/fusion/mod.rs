//! Runtime-rank expression fusion implemented by the WGPU provider.

mod dispatch;
mod source;

pub use dispatch::WgpuFusionOps;
pub use source::WgpuFusionScalar;
