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

use hephaestus_core::{ComputeDevice, DecompositionOps};

mod cholesky;
mod common;
mod lu;
mod qr;
mod spectral;
mod symmetric;

/// Run every decomposition clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_decomposition_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    lu::lu_pivots_solves_and_reconstructs(device, ops);
    lu::lu_rejects_a_non_square_operand(device, ops);
    cholesky::cholesky_factors_an_spd_matrix(device, ops);
    cholesky::cholesky_rejects_an_indefinite_matrix(device, ops);
    qr::qr_solves_a_consistent_least_squares_system(device, ops);
    qr::col_piv_qr_reveals_rank_and_solves(device, ops);
    spectral::symmetric_eigen_diagonalizes_exactly(device, ops);
    spectral::svd_recovers_exact_spectra(device, ops);
    symmetric::symmetric_indefinite_factorizations_reconstruct(device, ops);
    spectral::general_spectral_reductions_hold_their_invariants(device, ops);
    lu::full_piv_lu_reveals_rank_and_reconstructs(device, ops);
    assert_leto_differential_contract(device, ops);
    common::identity_factorizations_are_exact(device, ops);
}

/// The closed set of decomposition operations with an independent Leto oracle.
///
/// The remaining decomposition operations are checked through analytical
/// invariants above. Keeping this set explicit makes the differential coverage
/// auditable while one parameterized clause owns its execution for every
/// backend.
#[derive(Clone, Copy)]
enum LetoDifferential {
    Lu,
    Cholesky,
    Qr,
    ColPivQr,
    FullPivLu,
    Udu,
    BunchKaufman,
    SymmetricEigen,
    Eigenvalues,
}

const LETO_DIFFERENTIALS: [LetoDifferential; 9] = [
    LetoDifferential::Lu,
    LetoDifferential::Cholesky,
    LetoDifferential::Qr,
    LetoDifferential::ColPivQr,
    LetoDifferential::FullPivLu,
    LetoDifferential::Udu,
    LetoDifferential::BunchKaufman,
    LetoDifferential::SymmetricEigen,
    LetoDifferential::Eigenvalues,
];

/// Runs the complete parameterized differential clause against the Leto
/// reference for every backend.
fn assert_leto_differential_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    for case in LETO_DIFFERENTIALS {
        match case {
            LetoDifferential::Lu => lu::lu_matches_the_leto_reference(device, ops),
            LetoDifferential::Cholesky => {
                cholesky::cholesky_matches_the_leto_reference(device, ops)
            }
            LetoDifferential::Qr => qr::qr_matches_the_leto_reference(device, ops),
            LetoDifferential::ColPivQr => qr::col_piv_qr_matches_the_leto_reference(device, ops),
            LetoDifferential::FullPivLu => lu::full_piv_lu_matches_the_leto_reference(device, ops),
            LetoDifferential::Udu => symmetric::udu_matches_the_leto_reference(device, ops),
            LetoDifferential::BunchKaufman => {
                symmetric::bunch_kaufman_matches_the_leto_reference(device, ops)
            }
            LetoDifferential::SymmetricEigen => {
                spectral::symmetric_eigen_matches_the_leto_reference(device, ops);
            }
            LetoDifferential::Eigenvalues => {
                spectral::eigenvalues_match_the_leto_reference(device, ops)
            }
        }
    }
}
