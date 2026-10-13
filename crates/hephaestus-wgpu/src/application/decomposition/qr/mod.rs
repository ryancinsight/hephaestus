//! GPU-resident QR decomposition via Householder reflectors.
//!
//! Computes **A** = **Q R** where **Q** is orthogonal and **R** is
//! upper-triangular.
//!
//! Two entry points are provided:
//!
//! - [`qr_decompose`] — full host delegation (panel + trailing on CPU).
//! - [`qr_decompose_blocked`] — shape-selected hybrid algorithm. Matrices up to
//!   four panels wide or below 2048 rows use one host factorization; taller,
//!   wider matrices factor panels on the CPU, apply wide trailing Householder
//!   updates on the GPU, and finish the final at-most-one-block tail on the
//!   CPU after a paired readback.
//!
//! # Mathematical Foundations
//!
//! ## Theorem — Householder QR Factorization
//!
//! Every **A** ∈ ℝᵐˣⁿ with *m* ≥ *n* factors as **A** = **Q R** where
//! **Q** ∈ ℝᵐˣᵐ is orthogonal and **R** ∈ ℝᵐˣⁿ is upper-triangular.
//!
//! **Proof.** At step *k*, the Householder reflector **Hₖ** = **I** − βₖ
//! **vₖ vₖ**ᵀ zeros the entries below the diagonal of column *k*.
//! **Hₖ** is orthogonal (**Hₖ**ᵀ = **Hₖ** and **Hₖ²** = **I**).
//! After *n* steps, **Hₙ ⋯ H₁ A** = **R** is upper-triangular, so
//! **Q** = **H₁ᵀ ⋯ Hₙᵀ** = **H₁ ⋯ Hₙ** (each reflector is symmetric). ∎
//!
//! ## Blocked QR with GPU Trailing Application
//!
//! For large *m*, the dominant cost is applying the *b* Householder
//! reflectors from each panel to the trailing *m × (n−k−b)* submatrix.
//! Each application costs O(m(n−k)) flops — b applications per panel gives
//! O(b·m·(n−k)) — and is embarrassingly parallel across columns.
//!
//! **Theorem (Blocked QR complexity).** For *m × n* with block size *b*,
//! the total flop count is 2n²(m − n/3), identical to unblocked QR.  The
//! blocked variant improves performance by:
//! (a) moving the O(b·(m−k)·(n−k)) trailing Householder application to
//!     the GPU, and
//! (b) improving CPU cache locality for the O(b²·(m−k)) panel operations.
//!
//! **Proof.** Each block iteration costs:
//! - Panel factor: 2b²(m−k) − 2b³/3
//! - Trailing apply: 2b(m−k)(n−k−b)  (b rank-1 updates of width n−k−b)
//!
//! Summing over all ⌈n/b⌉ blocks recovers 2n²(m − n/3) total flops. ∎

mod accumulate;
mod api;
mod householder;
mod panel;
#[cfg(test)]
mod tests;

pub use api::{qr_decompose, qr_decompose_blocked};
pub use panel::GpuQrDecomposition;
