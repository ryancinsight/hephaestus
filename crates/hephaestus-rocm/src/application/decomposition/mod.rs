//! Device-resident dense decompositions for the ROCm backend.
//!
//! The decomposition feature is additive and mirrors the common Cholesky, LU,
//! complete-pivoted LU, QR, and column-pivoted QR surfaces exposed by the CUDA
//! and wgpu backends. Factorization is
//! executed by HIP kernels; returned values retain host copies only for the
//! existing scalar solve and inspection contracts.

mod cholesky;
mod col_piv_qr;
mod full_piv_lu;
mod lu;
mod qr;
mod validate;

pub use cholesky::{GpuCholesky, cholesky_decompose, cholesky_decompose_blocked};
pub use col_piv_qr::{GpuColPivQrDecomposition, col_piv_qr, col_piv_qr_blocked};
pub use full_piv_lu::{GpuFullPivLuDecomposition, full_piv_lu, full_piv_lu_blocked};
pub use lu::{GpuLuDecomposition, lu_decompose, lu_decompose_blocked};
pub use qr::{GpuQrDecomposition, qr_decompose, qr_decompose_blocked};
