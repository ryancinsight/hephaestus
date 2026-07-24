//! Device-resident dense decompositions for the ROCm backend.
//!
//! The decomposition feature is additive and mirrors the common Cholesky
//! surface exposed by the CUDA and wgpu backends. Factorization is executed by
//! HIP kernels; the returned value also retains a host copy of the factor for
//! the existing solve, determinant, and inverse contract.

mod cholesky;
mod validate;

pub use cholesky::{GpuCholesky, cholesky_decompose, cholesky_decompose_blocked};
