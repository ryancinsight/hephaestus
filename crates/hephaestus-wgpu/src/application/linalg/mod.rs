//! GPU-resident linear algebra operations.
//!
//! Provides optimized matrix multiplication, matrix functions, vector dot
//! products, matrix trace, and vector/matrix norms (L1, L2, Max) mapped to GPU
//! compute dispatches.

#[cfg(test)]
mod identity_contract;
mod map_reduction;
#[cfg(test)]
mod matrix_properties_contract;
pub use map_reduction::{
    L2NormScalar, PreparedDot, PreparedL2Norm, dot, norm_l1, norm_l2, norm_max, prepare_dot,
    prepare_norm_l2, trace,
};

mod dispatch;
mod identity;
mod kron;
mod layout;
mod matmul;
mod matpow;
mod ops;
mod properties;
mod scalars;

#[cfg(feature = "decomposition")]
pub(crate) use identity::device_identity;
pub use kron::{kron, kron_into};
pub use matmul::{batched_matmul, batched_matmul_into, matmul, matmul_into};
pub use matpow::matpow;
#[cfg(feature = "decomposition")]
pub use ops::MatrixDecompose;
pub use ops::{
    AsGpuMatrixOperand, MatrixFunction, MatrixNorm, MatrixProduct, MatrixProperties, MatrixSolve,
};
pub use properties::{det, matrix_rank, matrix_rank_with_tolerance};
#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub use properties::{matexp, pinv};
pub use scalars::{MatmulZero, MatrixIdentityScalar, MatrixRankScalar};
