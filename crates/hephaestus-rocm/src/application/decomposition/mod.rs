//! Device-resident dense decompositions for the ROCm backend.
//!
//! The decomposition feature is additive and mirrors the common Cholesky, LU,
//! complete-pivoted LU, QR, column-pivoted QR, bidiagonalization, SVD, UDU,
//! and Bunch–Kaufman
//! surfaces exposed by the CUDA and wgpu backends. Native HIP kernels own the
//! pivoted factorization paths; the spectral paths use the shared Leto provider
//! boundary and upload typed device factors.

mod bidiagonal;
mod bunch_kaufman;
mod cholesky;
mod col_piv_qr;
mod eigen;
mod full_piv_lu;
mod hessenberg;
mod lu;
mod qr;
mod schur;
mod svd;
mod udu;
mod validate;

pub use bidiagonal::{GpuBidiagonalDecomposition, bidiagonalize};
pub use bunch_kaufman::{GpuBunchKaufmanDecomposition, bunch_kaufman};
pub use cholesky::{GpuCholesky, cholesky_decompose, cholesky_decompose_blocked};
pub use col_piv_qr::{GpuColPivQrDecomposition, col_piv_qr, col_piv_qr_blocked};
pub use eigen::{
    GpuSymmetricEigenDecomposition, eigenvalues, symmetric_eigen_jacobi,
    symmetric_eigenvalues_jacobi,
};
pub use full_piv_lu::{GpuFullPivLuDecomposition, full_piv_lu, full_piv_lu_blocked};
pub use hessenberg::{GpuHessenbergDecomposition, hessenberg};
pub use lu::{GpuLuDecomposition, lu_decompose, lu_decompose_blocked};
pub use qr::{GpuQrDecomposition, qr_decompose, qr_decompose_blocked};
pub use schur::{GpuRealSchur, schur};
pub use svd::{GpuSvdDecomposition, singular_values, svd_decompose, svd_rank_revealing};
pub use udu::{GpuUduDecomposition, udu_decompose};
