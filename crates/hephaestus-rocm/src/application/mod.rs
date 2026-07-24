//! ROCm application-layer compute operations.

/// Rank-2 axis reductions over leto layouts.
pub mod axis_reduction;
/// Device-resident dense matrix decompositions.
#[cfg(feature = "decomposition")]
pub mod decomposition;
pub mod elementwise;
/// Rank-2 matrix multiplication over strided layouts.
pub mod linalg;
pub(crate) mod pipeline;
/// Seeded host-delegated random initializers.
pub mod random;
/// Contiguous multi-pass tree reductions.
pub mod reduction;
/// Rank-2 prefix and suffix scans over strided layouts.
pub mod scan;
/// Device-resident CSR sparse matrix products.
pub mod sparse;
/// Backend-neutral multi-storage kernel dispatch.
pub mod storage_kernel;
/// Backend-neutral authored-kernel command streams.
pub mod stream;
/// Layout-aware operand descriptors.
pub mod strided;
/// Rank-≤4 layout-aware elementwise operations.
pub mod strided_elementwise;
