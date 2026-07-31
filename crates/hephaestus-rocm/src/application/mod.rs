//! ROCm application-layer compute operations.

/// Rank-2 axis reductions over leto layouts.
pub mod axis_reduction;
#[cfg(all(feature = "rocm", target_os = "linux"))]
/// Device-neutral axis-reduction seam implementation.
pub mod axis_reduction_seam;
/// Native HIP regular and transposed convolution.
#[cfg(all(feature = "rocm", target_os = "linux"))]
pub mod convolution;
/// Device-resident dense matrix decompositions.
#[cfg(feature = "decomposition")]
pub mod decomposition;
pub mod elementwise;
#[cfg(all(feature = "rocm", target_os = "linux"))]
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
#[cfg(all(feature = "rocm", target_os = "linux"))]
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Rank-2 matrix multiplication over strided layouts.
pub mod linalg;
pub(crate) mod pipeline;
/// Reusable rank-2 axis reduction plans.
pub mod prepared_axis_reduction;
/// Reusable dot-product and L2-norm map-reduction plans.
pub mod prepared_map_reduction;
/// Reusable multi-pass scalar reduction plans.
pub mod prepared_reduction;
/// Seeded host-delegated random initializers.
pub mod random;
/// Contiguous multi-pass tree reductions.
pub mod reduction;
/// Rank-2 prefix and suffix scans over strided layouts.
pub mod scan;
/// Device-neutral scan seam implementation.
pub mod scan_seam;
/// Device-resident CSR sparse matrix products.
pub mod sparse;
/// Two-dimensional Laplacian stencil kernels.
pub mod stencil;
/// Backend-neutral multi-storage kernel dispatch.
pub mod storage_kernel;
/// Backend-neutral authored-kernel command streams.
pub mod stream;
/// Layout-aware operand descriptors.
pub mod strided;
/// Rank-≤4 layout-aware elementwise operations.
pub mod strided_elementwise;
/// Dense vector recurrences and prepared reductions.
pub mod vector;
/// Volume ray-integral kernels.
pub mod volume;
