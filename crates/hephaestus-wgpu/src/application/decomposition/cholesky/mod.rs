//! GPU-resident Cholesky decomposition.
//!
//! Computes **A** = **L** **L**ᵀ for symmetric positive-definite matrices.
//!
//! Two entry points are provided:
//!
//! - [`cholesky_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`cholesky_decompose_blocked`] — blocked algorithm where panel
//!   factorization runs on the CPU but the O(n³) trailing SYRK update
//!   (`A₂₂ -= L₂₁ L₂₁ᵀ`) runs on the GPU via a dedicated compute kernel.

mod blocked;
mod factor;
mod solve;
mod syrk;

pub use blocked::cholesky_decompose_blocked;
pub use factor::{GpuCholesky, cholesky_decompose};
