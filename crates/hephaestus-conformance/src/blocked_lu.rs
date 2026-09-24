//! Contract clause for the backend-abstracted blocked LU loop (ADR 0003,
//! SUBSTRATE-003).
//!
//! The blocked `*_decompose_blocked` entry points of the wgpu and CUDA
//! backends share one host-orchestration loop: the panel iteration, the CPU
//! panel factorisation, the permutation and sign bookkeeping, and the
//! trailing-update schedule all live in
//! [`blocked_lu`](hephaestus_core::blocked_lu), generic over
//! [`BlockedDecompositionBackend`](hephaestus_core::BlockedDecompositionBackend).
//! Only the device startup copy, the region gather/scatter, and the trailing
//! `A₂₂ -= L₂₁·U₁₂` kernel are backend-specific.
//!
//! Each backend's `tests/contract.rs` used to carry its own copy of the
//! blocked-LU differential fixtures and assertions, so the shared loop was
//! re-verified per backend in slightly different terms. This clause owns the
//! fixtures once; a backend runs it by instantiating, the same shape as the
//! other seam clauses.
//!
//! The clause drives the loop directly rather than a backend's public
//! `lu_decompose_blocked` wrapper, because the wrapper adds only operand
//! validation and the `leto_ops::LuDecomposition` assembly the clause
//! reconstructs from the loop's own host factors.

use hephaestus_core::test_support::assert_rejects;
use hephaestus_core::{
    BlockedDecompositionBackend, BlockedLuFactors, PanelRegion, Result, blocked_lu,
};

/// Backward-stability slack for the differential solve comparisons.
///
/// The loop runs Leto's own partial-pivoting panel elimination on the host,
/// so the two solves differ only in the trailing `A₂₂ -= L₂₁·U₁₂` update,
/// which each backend evaluates in `f32` on the device in its own
/// accumulation order. On these strictly diagonally dominant fixtures
/// (`κ∞ ≤ 1.03`, growth `ρ ≤ 2`) the classical bound is
/// `2·c(n)·ε·κ∞·‖x‖∞` with `c(n) ≤ 3n` (Higham, *Accuracy and Stability of
/// Numerical Algorithms*, ch. 9); the clauses assert `12·n·ε·‖x‖∞`, roughly
/// 2× that bound.
const SOLVE_SLACK: f32 = 12.0;

/// Run every blocked-LU clause against one backend.
///
/// `block_size` is the backend's panel width (its `LU_BLOCK_SIZE`), which
/// fixes the block-boundary fixture's dimension at `block_size + 2` so the
/// loop always crosses at least one panel boundary.
///
/// # Panics
///
/// Panics with the violated clause when the shared loop does not satisfy the
/// contract. A backend calls this from a test that has already acquired a
/// device.
pub fn assert_blocked_lu_contract<B>(backend: &B, block_size: usize)
where
    B: BlockedDecompositionBackend,
{
    identity_factors_are_exact(backend, block_size);
    solves_a_known_system(backend, block_size);
    matches_the_leto_reference_across_a_block_boundary(backend, block_size);
    rejects_a_singular_matrix(backend, block_size);
}

/// Stage a dense `n × n` host matrix on the device and run the shared loop.
///
/// The compact scratch buffer is sized `n × n` — the largest region any
/// single `write_region` call stages — and the loop allocates its own panel
/// scratch internally.
fn factor<B>(
    backend: &B,
    host: &[f32],
    n: usize,
    block_size: usize,
) -> Result<BlockedLuFactors<B::Buffer>>
where
    B: BlockedDecompositionBackend,
{
    let factors = backend.alloc(n * n)?;
    let scratch = backend.alloc(n * n)?;
    let region = PanelRegion {
        stride: n,
        row0: 0,
        col0: 0,
        rows: n,
        cols: n,
    };
    backend.write_region(&factors, region, &scratch, host)?;
    blocked_lu(backend, factors, n, block_size.min(n))
}

/// Assemble the host-side decomposition from the loop's own bookkeeping,
/// exactly as a backend's `lu_decompose_blocked` wrapper does.
fn host_decomposition<B>(
    factors: &BlockedLuFactors<B::Buffer>,
    n: usize,
) -> leto_ops::LuDecomposition<f32>
where
    B: BlockedDecompositionBackend,
{
    leto_ops::LuDecomposition::from_raw_parts(
        leto::Array2::from_shape_vec([n, n], factors.host.clone()).expect("square factor matrix"),
        factors.perm.clone(),
        factors.sign,
    )
}

/// The independent Leto LU of the same dense matrix.
fn leto_reference(host: &[f32], n: usize) -> leto_ops::LuDecomposition<f32> {
    let matrix = leto::Array::from_shape_vec([n, n], host.to_vec()).expect("square fixture");
    leto_ops::lu_decompose(&matrix.view()).expect("leto LU")
}

/// Solve `decomposition · x = rhs` on the host and return `x`.
fn solve(decomposition: &leto_ops::LuDecomposition<f32>, rhs: &[f32], n: usize) -> Vec<f32> {
    let rhs = leto::Array::from_shape_vec([n], rhs.to_vec()).expect("right-hand side");
    let x = decomposition.solve(&rhs.view()).expect("LU solve");
    leto::Storage::as_slice(x.storage()).to_vec()
}

/// Factorizing the identity is exact: every step multiplies by 0 or 1, so
/// the packed factors, the permutation, and the sign admit exact equality.
fn identity_factors_are_exact<B>(backend: &B, block_size: usize)
where
    B: BlockedDecompositionBackend,
{
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let result = factor(backend, &identity, 2, block_size).expect("identity factorization");

    assert_eq!(
        result.host.as_slice(),
        &identity[..],
        "blocked LU: identity factors must be the identity"
    );
    assert_eq!(
        result.perm,
        vec![0, 1],
        "blocked LU: identity permutation must be the identity"
    );
    assert_eq!(result.sign, 1, "blocked LU: identity sign");

    let reference = leto_reference(&identity, 2);
    assert_eq!(
        reference.det(),
        1.0,
        "blocked LU: identity determinant is one"
    );
    assert_eq!(
        host_decomposition::<B>(&result, 2).det(),
        reference.det(),
        "blocked LU: identity determinant vs leto"
    );
}

/// `A = [[2,1],[4,3]]`, `b = [5,11]ᵀ` has the exact solution `x = [2,1]ᵀ`.
/// Every elimination and substitution step is dyadic, so the bound admits one
/// reciprocal-multiply rounding per step.
fn solves_a_known_system<B>(backend: &B, block_size: usize)
where
    B: BlockedDecompositionBackend,
{
    let a_host = [2.0f32, 1.0, 4.0, 3.0];
    let rhs = [5.0f32, 11.0];
    let result = factor(backend, &a_host, 2, block_size).expect("known-system factorization");

    let got = solve(&host_decomposition::<B>(&result, 2), &rhs, 2);
    let expected = solve(&leto_reference(&a_host, 2), &rhs, 2);
    let analytic = [2.0f32, 1.0];
    for index in 0..2 {
        let tolerance = 4.0 * f32::EPSILON * expected[index].abs();
        assert!(
            (got[index] - expected[index]).abs() <= tolerance,
            "blocked LU: solve x[{index}] = {} expected {}",
            got[index],
            expected[index]
        );
        assert!(
            (got[index] - analytic[index]).abs() <= tolerance,
            "blocked LU: solve x[{index}] = {} expected analytic {}",
            got[index],
            analytic[index]
        );
    }
}

/// The block-boundary fixture — strictly diagonally dominant with a forced
/// pivot swap at the first row and across the boundary — is factored by the
/// loop and compared against Leto on the determinant and the solve.
///
/// The determinant is compared exactly: it is `sign · Π Uᵢᵢ`, and both sides
/// assemble it from the same host panel elimination. The solve carries the
/// derived [`SOLVE_SLACK`] bound.
fn matches_the_leto_reference_across_a_block_boundary<B>(backend: &B, block_size: usize)
where
    B: BlockedDecompositionBackend,
{
    let n = block_size + 2;
    let mut a_host = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            a_host[row * n + col] = if row == col {
                n as f32 + 4.0
            } else {
                0.1 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    a_host[0] = 0.0;
    a_host[block_size * n + block_size] = 0.0;

    let result = factor(backend, &a_host, n, block_size).expect("block-boundary factorization");
    assert_eq!(
        result.perm.len(),
        n,
        "blocked LU: permutation length must equal the order"
    );

    let reference = leto_reference(&a_host, n);
    let decomposition = host_decomposition::<B>(&result, n);
    assert_eq!(
        decomposition.det(),
        reference.det(),
        "blocked LU: block-boundary determinant vs leto"
    );

    let rhs = vec![1.0f32; n];
    let got = solve(&decomposition, &rhs, n);
    let expected = solve(&reference, &rhs, n);
    let x_inf = expected
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    let bound = SOLVE_SLACK * n as f32 * f32::EPSILON * x_inf;
    for index in 0..n {
        assert!(
            (got[index] - expected[index]).abs() <= bound,
            "blocked LU: solve x[{index}] = {} expected {}",
            got[index],
            expected[index]
        );
    }
}

/// A zero pivot in the first panel is rejected with the panel
/// factorisation's located error, before any trailing update runs.
fn rejects_a_singular_matrix<B>(backend: &B, block_size: usize)
where
    B: BlockedDecompositionBackend,
{
    let singular = [0.0f32, 0.0, 0.0, 1.0];
    let result = factor(backend, &singular, 2, block_size);
    assert_rejects(
        &result,
        "kernel dispatch failed: LU panel factorisation failed: pivot column 0 is exactly zero",
    );
}
