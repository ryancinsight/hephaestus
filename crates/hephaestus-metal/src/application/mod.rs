//! Compute dispatch delegation to Metal.

/// Matrix decompositions.
#[cfg(feature = "decomposition")]
pub mod decomposition;
/// Elementwise compute dispatch.
pub mod elementwise;
/// Linear algebra operations.
pub mod linalg;
/// Reduction operations.
pub mod reduction;
/// Scan operations.
pub mod scan;
/// GPU-resident CSR sparse matrix operations.
pub mod sparse;
/// 2D Laplacian stencil delegation.
pub mod stencil;
/// Strided layout wrappers.
pub mod strided;
/// Volume ray-integral delegation.
pub mod volume;
