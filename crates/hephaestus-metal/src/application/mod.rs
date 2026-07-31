//! Compute dispatch delegation to Metal.

/// Zero-copy convolution delegation to Metal-selected WGPU.
pub mod convolution;
/// Matrix decompositions.
#[cfg(feature = "decomposition")]
pub mod decomposition;
/// Elementwise compute dispatch.
pub mod elementwise;
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Linear algebra operations.
pub mod linalg;
/// Fluent dense-matrix traits.
pub mod linalg_traits;
/// Reusable dot-product and L2-norm map-reduction plans.
pub mod prepared_map_reduction;
/// Seeded host-delegated random initializers.
pub mod random;
/// Reduction operations.
pub mod reduction;
/// Scan operations.
pub mod scan;
/// Device-neutral scan seam implementation.
pub mod scan_seam;
/// GPU-resident CSR sparse matrix operations.
pub mod sparse;
/// 2D Laplacian stencil delegation.
pub mod stencil;
/// Metal-selected storage-kernel dispatch.
pub mod storage_kernel;
/// Metal-selected authored-kernel command streams.
pub mod stream;
/// Strided layout wrappers.
pub mod strided;
/// Dense vector recurrence operations.
pub mod vector;
/// Volume ray-integral delegation.
pub mod volume;
