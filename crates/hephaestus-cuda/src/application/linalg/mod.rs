//! Linear algebra operations on the CUDA device.
//!
//! Two families with distinct dispatch strategies share this module:
//!
//! - `matmul` / `batched_matmul` (the `matmul` submodule): a bespoke tiled GPU
//!   kernel, authored as CUDA C and launched directly via `cuLaunchKernel`.
//! - `dot` / `trace` / `norm_l1` / `norm_l2` / `norm_max` (the `norms`
//!   submodule): fused strided map-reductions whose first pass reads the
//!   logical view directly and retains only workgroup partials.

mod kron;
mod layout;
mod matmul;
mod matpow;
mod matrix;
mod matrix_rank;
mod norms;
mod pinv_matexp;
#[cfg(test)]
mod word_tests;

pub use kron::{kron, kron_into};
pub use matmul::{batched_matmul, batched_matmul_into, matmul, matmul_into};
#[cfg(all(feature = "cuda", feature = "decomposition"))]
pub(crate) use matpow::device_identity;
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
pub use norms::{dot, norm_l1, norm_l2, norm_max, trace};
pub use pinv_matexp::{matexp, pinv};

use hephaestus_core::map_layout_err;
use layout::{GpuMatrixLayout, map_layout, validate_signed_address_range};
pub(crate) use layout::{to_i32, to_i64, to_u32};
