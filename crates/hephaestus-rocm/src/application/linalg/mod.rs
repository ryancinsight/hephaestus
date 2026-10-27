//! Strided linear algebra operations on the ROCm device.
//!
//! Matrix multiplication uses one HIP workgroup per 16×16 output tile and
//! shared-memory tiles along the contracted dimension. Host-side validation
//! preserves the same strided-layout and alias contract as the other GPU
//! backends. The batched form dispatches the batch dimension through grid-z
//! and supports singleton input-batch broadcasting. Map-reduction operations
//! use the same layout metadata for dot products, traces, and norms.
//! Kronecker products use the same strided rank-2 metadata and one output
//! thread per logical element.

mod batched_matmul;
mod kron;
mod layout;
mod matmul;
mod matpow;
mod matrix;
mod matrix_rank;
mod norms;
mod pinv_matexp;

pub use batched_matmul::{batched_matmul, batched_matmul_into};
pub use kron::{kron, kron_into};
pub use matmul::{matmul, matmul_into};
pub use matpow::{MatrixIdentityScalar, matpow};
#[cfg(feature = "decomposition")]
pub use matrix::MatrixDecompose;
pub use matrix::{
    AsGpuMatrixOperand, MatrixFunction, MatrixNorm, MatrixProduct, MatrixProperties, MatrixSolve,
};
pub use matrix_rank::{MatrixRankScalar, det, matrix_rank, matrix_rank_with_tolerance};
pub(crate) use norms::{
    DotMap, PreparedMapReduction, RetainedMapReduction, SquareMap,
    prepare_map_reduction_with_layouts,
};
pub use norms::{L2NormScalar, dot, norm_l1, norm_l2, norm_max, trace};
pub use pinv_matexp::{matexp, pinv};

use hephaestus_core::map_layout_err;
use layout::{GpuMatrixLayout, map_layout, to_i64};
pub(crate) use layout::{to_i32, to_u32};
