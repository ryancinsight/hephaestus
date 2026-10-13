//! Aggregate analytical and Leto differential decomposition contracts.

use super::{cholesky, identity, lu, qr, spectral, symmetric};
use hephaestus_core::{ComputeDevice, DecompositionOps};

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
    identity::identity_factorizations_are_exact(device, ops);
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
