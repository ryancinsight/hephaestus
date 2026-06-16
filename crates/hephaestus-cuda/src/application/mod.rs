//! Monomorphized compute dispatch over the CUDA device.

/// CUDA primitive type mappings.
pub mod cuda_type;
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
/// Layout-aware strided elementwise operations.
pub mod strided;

#[cfg(feature = "decomposition")]
/// Dense matrix decompositions (Cholesky, LU, QR) backed by leto-ops.
pub mod decomposition;
