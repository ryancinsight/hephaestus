//! ROCm application-layer compute operations.

pub mod elementwise;
pub(crate) mod pipeline;
/// Contiguous multi-pass tree reductions.
pub mod reduction;
