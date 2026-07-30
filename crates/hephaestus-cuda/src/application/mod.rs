//! Monomorphized compute dispatch over the CUDA device.

/// Native regular and transposed convolution operations.
pub mod convolution;
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
/// Contiguous elementwise operations.
pub mod elementwise;
/// Linear algebra operations (matmul, batch matmul, trace, dot, norms).
pub mod linalg;
/// Pipeline compilation and launch helpers.
pub mod pipeline;
/// Reusable rank-2 axis reduction plans.
pub mod prepared_axis_reduction;
/// Reusable dot-product and L2-norm map-reduction plans.
pub mod prepared_map_reduction;
/// Reusable multi-pass scalar reduction plans.
pub mod prepared_reduction;
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Multi-pass tree reductions.
pub mod reduction;
/// Device-neutral scan seam implementation.
pub mod scan_seam;
/// Prefix/suffix scan operations.
pub mod scan;
/// Two-dimensional Laplacian stencil kernels.
pub mod stencil;
/// Backend-neutral multi-storage kernel dispatch.
pub mod storage_kernel;
/// Backend-neutral command stream implementation for authored CUDA C kernels.
pub mod stream;
/// Layout-aware strided elementwise operations.
pub mod strided;
/// Dense vector recurrences and prepared reductions.
pub mod vector;
/// Volume ray-integral kernels.
pub mod volume;

#[cfg(feature = "decomposition")]
/// Dense matrix decompositions (Cholesky, LU, QR) backed by leto-ops.
pub mod decomposition;

/// Seeded host-delegated PRNG initializers.
pub mod random;
/// GPU Compressed Sparse Row (CSR) sparse matrix operations.
pub mod sparse;
