#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! Shared CPU-side panel factorisation routines for blocked decomposition,
//! plus backend-neutral operand validation.
//!
//! The panel routines operate on packed row-major `f32` slices and return the
//! bookkeeping data the blocked loop needs (permutation vectors, Householder
//! heads, β coefficients). The
//! [`validate_square_operand`](crate::domain::decomposition::validate_square_operand) and
//! [`require_dense_operand`](crate::domain::decomposition::require_dense_operand) validators
//! check a rank-2 [`leto::Layout`] once so
//! both backends share the same dispatch preconditions. All are used by both
//! the wgpu and CUDA backends.

pub mod blocked;

pub use blocked::{
    BlockedCholeskyBackend, BlockedCholeskyFactors, BlockedDecompositionBackend, BlockedLuFactors,
    BlockedQrBackend, BlockedQrFactors, PanelRegion, TrailingGemm, TrailingHh, TrailingSyrk,
    blocked_cholesky, blocked_lu, blocked_qr,
};

mod cholesky;
mod lu;
mod qr;
mod validate;

#[cfg(test)]
mod tests;

pub use cholesky::{factor_cholesky_panel, panel_cholesky_packed};
pub use lu::{factor_lu_panel, panel_lu_packed, split_packed_lu};
pub use qr::{apply_packed_qr_panel_left, panel_qr_packed};
pub use validate::{require_dense_operand, validate_square_operand};
