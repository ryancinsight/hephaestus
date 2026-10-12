//! Device-neutral dense decomposition seam (ADR 0042).
//!
//! One trait covers the core factorization trio — LU with partial pivoting,
//! Householder QR, and Cholesky — over rank-2 strided `f32` operands. The
//! scalar type is fixed at `f32` by every backend's decomposition kernels;
//! the scalar dimension enters here when the kernels ship it, not before.
//!
//! Result handles are exposed through oracle-minimal accessor traits: exactly
//! the capability consumers and the conformance oracles need (factor access,
//! pivots, determinants, solves). Backend-specific extras stay inherent on
//! the concrete handle types. The associated result types carry an `'op`
//! lifetime like the prepared forms, so a backend whose handle borrows
//! operand or workspace state implements without erasing that borrow;
//! today's handle-owning backends simply ignore it.

mod handles;
mod ops;

pub use handles::{
    BidiagonalHandle, BunchKaufmanHandle, CholeskyHandle, ColPivQrHandle, FullPivLuHandle,
    HessenbergHandle, LuHandle, QrHandle, SchurHandle, SvdHandle, SymmetricEigenHandle, UduHandle,
};
pub use ops::DecompositionOps;
