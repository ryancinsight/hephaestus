//! Monomorphized compute dispatch over the CUDA device.

/// Contiguous elementwise operations.
pub mod elementwise;
/// Linear algebra operations (matmul, batch matmul, trace, dot, norms).
pub mod linalg;
/// Pipeline compilation and launch helpers.
pub mod pipeline;
/// Multi-pass tree reductions.
pub mod reduction;
/// Prefix/suffix scan operations.
pub mod scan;
/// Backend-neutral multi-storage kernel dispatch.
pub mod storage_kernel;
/// Backend-neutral command stream implementation for authored CUDA C kernels.
pub mod stream;
/// Layout-aware strided elementwise operations.
pub mod strided;

#[cfg(feature = "decomposition")]
/// Dense matrix decompositions (Cholesky, LU, QR) backed by leto-ops.
pub mod decomposition;

/// Seeded host-delegated PRNG initializers.
pub mod random;
/// GPU Compressed Sparse Row (CSR) sparse matrix operations.
pub mod sparse;
