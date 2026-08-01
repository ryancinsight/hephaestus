//! Monomorphized compute dispatch over the wgpu device.

pub(crate) mod bindings;
/// Dense matrix decompositions (Cholesky, LU, QR).
#[cfg(feature = "decomposition")]
pub mod decomposition;
/// Elementwise binary kernels.
pub mod elementwise;
/// Linear algebra compute operations.
pub mod linalg;
pub(crate) mod pipeline;

/// Provider-owned finite-difference stencil operators.
pub mod stencil;

/// Native WGSL scaled dot-product attention.
pub mod attention;
/// Device-neutral axis-reduction seam implementation.
pub mod axis_reduction_seam;
/// Native WGSL regular and transposed convolution.
pub mod convolution;
/// Device-neutral decomposition seam implementation.
pub mod decomposition_seam;
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Runtime-parameter unary elementwise dispatch.
pub mod parameterized_elementwise;
pub(crate) mod prepared;
/// Seeded host-delegated PRNG initializers.
pub mod random;
/// Reduction compute operations.
pub mod reduction;
/// Prefix and suffix scan compute operations.
pub mod scan;
/// Device-neutral prefix/scan seam implementation.
pub mod scan_seam;
#[cfg(feature = "sparse")]
/// GPU Compressed Sparse Row (CSR) sparse matrix operations.
pub mod sparse;
/// Generic WGSL storage-kernel dispatch.
pub mod storage_kernel;
/// Backend-neutral command stream implementation for authored WGSL kernels.
pub mod stream;
/// Strided-layout-aware dispatch over leto layout metadata.
pub mod strided;
/// Volume ray-integral kernels (CT/dose ray-trace primitive).
/// Dense vector-operation seam implementation.
pub mod vector;
pub mod volume;
