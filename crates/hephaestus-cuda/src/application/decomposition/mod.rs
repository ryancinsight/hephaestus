//! Dense matrix decompositions for the CUDA backend.
//!
//! Provides Cholesky, LU, and QR decompositions that establish API parity
//! with [`leto_ops`] linear algebra. Panel factorization delegates to the
//! CPU (via leto-ops) while the result is stored on the device for
//! downstream GPU consumers.

pub(crate) mod validate;

pub mod bidiagonal;
pub mod bunch_kaufman;
pub mod cholesky;
pub mod col_piv_qr;
pub mod eigen;
pub mod full_piv_lu;
pub mod hessenberg;
pub mod lu;
pub mod qr;
pub mod schur;
pub mod svd;
pub mod udu;

pub use bidiagonal::{bidiagonalize, GpuBidiagonalDecomposition};
pub use bunch_kaufman::{bunch_kaufman, GpuBunchKaufmanDecomposition};
pub use cholesky::{cholesky_decompose, cholesky_decompose_blocked, GpuCholesky};
pub use col_piv_qr::{col_piv_qr, GpuColPivQrDecomposition};
pub use eigen::{
    eigenvalues, symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi,
    GpuSymmetricEigenDecomposition,
};
pub use full_piv_lu::{full_piv_lu, GpuFullPivLuDecomposition};
pub use hessenberg::{hessenberg, GpuHessenbergDecomposition};
pub use lu::{lu_decompose, lu_decompose_blocked, GpuLuDecomposition};
pub use qr::{qr_decompose, qr_decompose_blocked, GpuQrDecomposition};
pub use schur::{schur, GpuRealSchur};
pub use svd::{singular_values, svd_decompose, svd_rank_revealing, GpuSvdDecomposition};
pub use udu::{udu_decompose, GpuUduDecomposition};
