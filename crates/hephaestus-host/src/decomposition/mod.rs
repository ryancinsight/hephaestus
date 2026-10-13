//! Leto as a decomposition-seam implementor (ADR 0046 / ADR 0039 §3).
//!
//! [`HostDecompositionOps`] adapts leto-ops' fifteen decomposition entry
//! points onto [`DecompositionOps<HostDevice>`](hephaestus_core::DecompositionOps), so the CPU substrate joins
//! the same role trait the GPU backends implement and the conformance
//! suite's decomposition clauses run on the host pair. Factor storage
//! conventions follow the seam contracts pinned by the clause suite
//! (packed LU factors, m×n row-major `R`, column-convention singular and
//! eigen vectors, dense block-diagonal Bunch–Kaufman `D`, length-`n` UDU
//! diagonal).

mod basic;
mod helpers;
mod ops;
mod orthogonal;
mod pivoted;
mod spectral;
mod symmetric;

pub use basic::{HostCholesky, HostLu, HostQr};
pub use helpers::HostDecompositionOps;
pub use orthogonal::{HostBidiagonal, HostHessenberg, HostSchur};
pub use pivoted::{HostColPivQr, HostFullPivLu};
pub use spectral::{HostSvd, HostSymmetricEigen};
pub use symmetric::{HostBunchKaufman, HostUdu};
