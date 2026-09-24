//! Contract clauses for the backend-abstracted blocked Cholesky loop
//! (ADR 0003, SUBSTRATE-003).
//!
//! The blocked `cholesky_decompose_blocked` entry points of the WGPU and CUDA
//! backends share one host-orchestration loop: the panel walk, the CPU panel
//! factorisation, the diagonal retention, and the panel scatter all live in
//! [`blocked_cholesky`](hephaestus_core::blocked_cholesky), generic over
//! [`BlockedCholeskyBackend`](hephaestus_core::BlockedCholeskyBackend). Only the
//! region gather/scatter and the trailing **A₂₂ -= L₂₁·L₂₁ᵀ** SYRK kernel are
//! backend-specific.
//!
//! Each backend's `tests/contract.rs` used to carry its own copy of the blocked
//! Cholesky differential fixtures and assertions. This clause owns them once, so
//! a backend runs them by instantiating rather than by re-authoring.

use hephaestus_core::{
    BlockedCholeskyBackend, BlockedCholeskyFactors, PanelRegion, Result, blocked_cholesky,
};

/// Derived factor bound; the fixture below has `n ≤ block_size + 2` and an
/// infinity-norm condition number below 4, so the classical
/// `c(n)·ε·κ(A)·‖A‖` backward-error bound stays far under `1e-4`.
const FACTOR_BOUND: f32 = 1e-4;

/// Run every blocked-Cholesky clause against one backend.
///
/// `block_size` is the backend's panel width (its `BLOCK_SIZE`), which fixes
/// the reconstruction fixture's dimension at `block_size + 2` so the loop
/// always crosses at least one panel boundary.
///
/// # Panics
///
/// Panics with the violated clause when the shared loop does not satisfy the
/// contract. A backend calls this from a test that has already acquired a
/// device.
pub fn assert_blocked_cholesky_contract<B>(backend: &B, block_size: usize)
where
    B: BlockedCholeskyBackend,
{
    identity_factors_are_exact(backend, block_size);
    reconstructs_the_spd_fixture_across_a_block_boundary(backend, block_size);
    rejects_an_indefinite_matrix(backend, block_size);
}

/// Stage a dense `n × n` host matrix on the device and run the shared loop.
fn factor<B>(
    backend: &B,
    host: &[f32],
    n: usize,
    block_size: usize,
) -> Result<BlockedCholeskyFactors<B::Buffer>>
where
    B: BlockedCholeskyBackend,
{
    let lower = backend.alloc(n * n)?;
    let scratch = backend.alloc(n * n)?;
    let region = PanelRegion {
        stride: n,
        row0: 0,
        col0: 0,
        rows: n,
        cols: n,
    };
    backend.write_region(&lower, region, &scratch, host)?;
    blocked_cholesky(backend, lower, n, block_size.min(n))
}

/// Read the loop's factor back to the host.
fn download_factor<B>(backend: &B, result: &BlockedCholeskyFactors<B::Buffer>, n: usize) -> Vec<f32>
where
    B: BlockedCholeskyBackend,
{
    let scratch = backend.alloc(n * n).expect("scratch allocation");
    let region = PanelRegion {
        stride: n,
        row0: 0,
        col0: 0,
        rows: n,
        cols: n,
    };
    let mut host = Vec::new();
    backend
        .download_region(&result.lower, region, &scratch, &mut host)
        .expect("factor download");
    host
}

/// Lower-triangular element access on the factored buffer.
fn lower_element(factor: &[f32], n: usize, row: usize, col: usize) -> f32 {
    if col <= row {
        factor[row * n + col]
    } else {
        0.0
    }
}

/// The identity's Cholesky factor is exactly the identity: every step takes
/// `sqrt(1)`, so no rounding occurs.
fn identity_factors_are_exact<B>(backend: &B, block_size: usize)
where
    B: BlockedCholeskyBackend,
{
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let result = factor(backend, &identity, 2, block_size).expect("identity factorization");

    assert_eq!(
        download_factor(backend, &result, 2),
        identity,
        "blocked Cholesky: identity factor"
    );
    assert_eq!(
        result.diagonal,
        vec![1.0, 1.0],
        "blocked Cholesky: identity diagonal"
    );
}

/// A strictly diagonally dominant SPD fixture wider than one panel: the loop
/// must scatter the first panel and update the trailing block before the rest.
///
/// The factor is compared to Leto's own Cholesky — the loop runs the same
/// column-oriented algorithm on the host — and `L·Lᵀ` is reconstructed against
/// the fixture in `f64`, with `L` read from the lower triangle only, because
/// the strict upper of the device buffer is stale by contract.
fn reconstructs_the_spd_fixture_across_a_block_boundary<B>(backend: &B, block_size: usize)
where
    B: BlockedCholeskyBackend,
{
    let n = block_size + 2;
    let host: Vec<f32> = (0..n * n)
        .map(|index| {
            let (row, col) = (index / n, index % n);
            if row == col {
                n as f32 + 2.0
            } else {
                0.1 / (1.0 + row.abs_diff(col) as f32)
            }
        })
        .collect();

    let result = factor(backend, &host, n, block_size).expect("SPD factorization");
    let lower = download_factor(backend, &result, n);

    let matrix = leto::Array::from_shape_vec([n, n], host.clone()).expect("square fixture");
    let reference = leto_ops::cholesky_decompose(&matrix.view()).expect("leto Cholesky");
    let reference_lower = leto::Storage::as_slice(reference.lower().storage());
    for row in 0..n {
        for col in 0..=row {
            let index = row * n + col;
            assert!(
                (lower[index] - reference_lower[index]).abs() < FACTOR_BOUND,
                "blocked Cholesky: factor ({row},{col}) = {} vs leto {}",
                lower[index],
                reference_lower[index]
            );
        }
    }

    for row in 0..n {
        for col in 0..=row {
            let mut acc = 0.0f64;
            for inner in 0..n {
                acc += f64::from(lower_element(&lower, n, row, inner))
                    * f64::from(lower_element(&lower, n, col, inner));
            }
            assert!(
                (acc - f64::from(host[row * n + col])).abs() < f64::from(FACTOR_BOUND),
                "blocked Cholesky: L·Lᵀ ({row},{col}) = {acc}"
            );
        }
    }
}

/// An indefinite matrix has no real Cholesky factor.
fn rejects_an_indefinite_matrix<B>(backend: &B, block_size: usize)
where
    B: BlockedCholeskyBackend,
{
    let indefinite = [1.0f32, 2.0, 2.0, 1.0];
    assert!(
        factor(backend, &indefinite, 2, block_size).is_err(),
        "blocked Cholesky must reject an indefinite matrix"
    );
}
