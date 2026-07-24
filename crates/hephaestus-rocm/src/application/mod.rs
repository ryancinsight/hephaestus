//! ROCm application-layer compute operations.

/// Rank-2 axis reductions over leto layouts.
pub mod axis_reduction;
pub mod elementwise;
pub(crate) mod pipeline;
/// Contiguous multi-pass tree reductions.
pub mod reduction;
/// Rank-2 prefix and suffix scans over strided layouts.
pub mod scan;
/// Layout-aware operand descriptors.
pub mod strided;
