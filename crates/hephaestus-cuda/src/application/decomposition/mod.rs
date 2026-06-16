//! Dense matrix decompositions for the CUDA backend.
//!
//! Provides Cholesky, LU, and QR decompositions that establish API parity
//! with [`leto_ops`] linear algebra. Panel factorization delegates to the
//! CPU (via leto-ops) while the result is stored on the device for
//! downstream GPU consumers.

pub(crate) mod validate;

pub mod cholesky;
pub mod lu;
pub mod qr;

pub use cholesky::{cholesky_decompose, GpuCholesky};
pub use lu::{lu_decompose, GpuLuDecomposition};
pub use qr::{qr_decompose, GpuQrDecomposition};
