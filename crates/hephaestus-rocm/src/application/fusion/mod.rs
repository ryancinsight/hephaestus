//! Runtime-rank expression fusion implemented by the ROCm provider.

mod dispatch;
mod source;

pub use dispatch::RocmFusionOps;
pub use source::RocmFusionScalar;
