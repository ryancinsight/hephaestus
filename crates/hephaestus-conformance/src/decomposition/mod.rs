//! Contract clauses for the device-neutral decomposition seam (ADR 0042).
//!
//! Each clause is generic over the device and the seam, so every backend
//! runs the same assertions against analytically known factorizations.
//!
//! ## Tolerance derivation
//!
//! For an order-`n` factorization in `f32` (`ε = 2⁻²³ ≈ 1.2e-7`), the
//! classical backward-error bounds give reconstruction and solve errors of
//! order `c(n)·ε·κ(A)` for modest constants `c(n)` (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, chs. 9–10). The fixtures here have
//! `n ≤ 4` and infinity-norm condition numbers below 32 by construction,
//! so `c(n)·ε·κ ≤ 3²·1.2e-7·32 ≈ 3.5e-4`; the clauses assert `1e-3`,
//! roughly 3× that bound, and reconstruction runs in `f64` on the host so
//! the comparison itself adds no `f32` error.

mod cholesky;
mod contract;
mod identity;
mod lu;
mod qr;
mod spectral;
mod symmetric;
mod tolerance;

pub use contract::assert_decomposition_contract;
