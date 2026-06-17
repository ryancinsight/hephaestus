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
/// Strided layout wrappers.
pub mod strided;
