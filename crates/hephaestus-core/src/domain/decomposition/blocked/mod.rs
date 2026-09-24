//! Backend-abstracted blocked decomposition loops.
//!
//! The blocked `*_decompose_blocked` entry points of the accelerator backends
//! share their host orchestration: the panel loop, the CPU panel factorization,
//! the permutation/sign bookkeeping, and the per-panel region index math. Only
//! the device operation *kinds* are backend-specific — the startup copy, the
//! compact region gather/scatter between device and host, and the
//! decomposition's trailing-matrix kernel. This module hoists the loops into
//! generic bodies over backend traits (ADR 0003):
//!
//! - [`blocked_lu`] over [`BlockedDecompositionBackend`], whose trailing update
//!   is the GEMM **C -= A·B**;
//! - [`blocked_qr`] over [`BlockedQrBackend`], which adds a reflector-metadata
//!   buffer and the trailing Householder application;
//! - [`blocked_cholesky`] over [`BlockedCholeskyBackend`], whose trailing
//!   update is the SYRK **A₂₂ -= L₂₁·L₂₁ᵀ**.
//!
//! QR and Cholesky extend [`BlockedDecompositionBackend`] with their own traits
//! rather than adding methods or associated types to it: either addition is a
//! breaking change for every implementor, including the backends whose entry
//! point has not migrated to the shared loop yet.
//!
//! Each loop owns all host bookkeeping; a backend implements the traits by
//! wrapping its existing region-transfer and trailing-kernel functions. The
//! compact device transfer buffer is allocated once above the loop and passed
//! through the region calls, so a backend that stages through it (wgpu) reuses
//! one device allocation per call instead of allocating per panel; a backend
//! that transfers via pinned host staging (CUDA) ignores it.

mod backend;
mod cholesky;
mod lu;
mod qr;

#[cfg(test)]
mod tests;

pub use backend::{BlockedDecompositionBackend, PanelRegion, TrailingGemm};
pub use cholesky::{
    BlockedCholeskyBackend, BlockedCholeskyFactors, TrailingSyrk, blocked_cholesky,
};
pub use lu::{BlockedLuFactors, blocked_lu};
pub use qr::{BlockedQrBackend, BlockedQrFactors, TrailingHh, blocked_qr};
