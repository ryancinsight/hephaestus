//! Contract clauses for the backend-abstracted blocked QR loop (ADR 0003,
//! SUBSTRATE-003).
//!
//! The blocked `qr_decompose_blocked` entry points of the WGPU and CUDA
//! backends share one host-orchestration loop: the panel walk, the CPU panel
//! factorisation, the reflector packing, the sub-diagonal zeroing, and the
//! final whole-matrix gather all live in
//! [`blocked_qr`](hephaestus_core::blocked_qr), generic over
//! [`BlockedQrBackend`](hephaestus_core::BlockedQrBackend). Only the region
//! gather/scatter, the reflector-metadata buffer, and the trailing Householder
//! application are backend-specific.
//!
//! Each backend's `tests/contract.rs` used to carry its own copy of the blocked
//! QR differential fixtures and assertions. This clause owns them once.
//!
//! The oracle is a host reconstruction: the loop returns the packed matrix, the
//! Householder heads and the `β` coefficients, so re-applying the reflectors to
//! **R** in reverse order rebuilds **A** without another device round trip.

use hephaestus_core::{BlockedQrBackend, BlockedQrFactors, PanelRegion, Result, blocked_qr};

/// Backward-error slack for the reconstruction comparisons.
///
/// Re-applying `n` Householder reflectors in `f32` rounds at each inner product
/// and each rank-1 update, so the rebuilt **A** differs from the fixture by
/// `c·n·ε·‖A‖∞` with a modest constant. The clause asserts
/// `12·n·ε·‖A‖∞`, the classical bound with the usual slack.
const RECONSTRUCTION_SLACK: f32 = 12.0;

/// Run every blocked-QR clause against one backend.
///
/// `block_size` is the backend's panel width (its `QR_BLOCK_SIZE`), which fixes
/// the reconstruction fixture's dimension at `block_size + 2` so the loop
/// always crosses at least one panel boundary.
///
/// # Panics
///
/// Panics with the violated clause when the shared loop does not satisfy the
/// contract. A backend calls this from a test that has already acquired a
/// device.
pub fn assert_blocked_qr_contract<B>(backend: &B, block_size: usize)
where
    B: BlockedQrBackend,
{
    identity_factor_reconstructs_the_identity(backend, block_size);
    reconstructs_the_fixture_across_a_block_boundary(backend, block_size);
    rectangular_r_matches_the_leto_reference(backend, block_size);
}

/// Stage a dense `m × n` host matrix on the device and run the shared loop.
fn factor<B>(
    backend: &B,
    host: &[f32],
    m: usize,
    n: usize,
    block_size: usize,
) -> Result<BlockedQrFactors<B::Buffer>>
where
    B: BlockedQrBackend,
{
    let work = backend.alloc(m * n)?;
    let scratch = backend.alloc(m * n)?;
    let region = PanelRegion {
        stride: n,
        row0: 0,
        col0: 0,
        rows: m,
        cols: n,
    };
    backend.write_region(&work, region, &scratch, host)?;
    blocked_qr(backend, work, m, n, block_size.min(n))
}

/// Download the factored matrix and assert its strict lower triangle is zero.
///
/// The loop scatters each panel with its sub-diagonal cleared, so **R**'s shape
/// is a contract, not a convention each backend keeps privately.
fn assert_upper_triangular<B>(
    backend: &B,
    result: &BlockedQrFactors<B::Buffer>,
    m: usize,
    n: usize,
) -> Vec<f32>
where
    B: BlockedQrBackend,
{
    let scratch = backend.alloc(m * n).expect("scratch allocation");
    let region = PanelRegion {
        stride: n,
        row0: 0,
        col0: 0,
        rows: m,
        cols: n,
    };
    let mut host = Vec::new();
    backend
        .download_region(&result.r, region, &scratch, &mut host)
        .expect("factor download");
    for row in 0..m {
        for col in 0..row.min(n) {
            assert_eq!(
                host[row * n + col],
                0.0,
                "blocked QR: R entry ({row},{col}) must vanish"
            );
        }
    }
    host
}

/// Re-apply the packed reflectors to **R** in reverse order to rebuild **A**.
///
/// `packed`'s strictly-lower entries are the packed reflector tails, so the
/// rebuild needs nothing but the loop's own host bookkeeping.
fn reconstruct<Buffer>(factors: &BlockedQrFactors<Buffer>, m: usize, n: usize) -> Vec<f32> {
    let packed = &factors.packed;
    let mut rebuilt = vec![0.0f32; m * n];
    for (row, line) in rebuilt.chunks_exact_mut(n).enumerate().take(n) {
        line[row..].copy_from_slice(&packed[row * n + row..(row + 1) * n]);
    }
    for j in (0..n).rev() {
        for col in 0..n {
            let mut dot = factors.heads[j] * rebuilt[j * n + col];
            for r in (j + 1)..m {
                dot += packed[r * n + j] * rebuilt[r * n + col];
            }
            let scale = factors.betas[j] * dot;
            rebuilt[j * n + col] -= scale * factors.heads[j];
            for r in (j + 1)..m {
                rebuilt[r * n + col] -= scale * packed[r * n + j];
            }
        }
    }
    rebuilt
}

/// Every entry of `rebuilt` matches `expected` within the fixture's bound.
fn assert_rebuilt(rebuilt: &[f32], expected: &[f32], magnitude: f32, n: usize, label: &str) {
    let bound = RECONSTRUCTION_SLACK * n as f32 * f32::EPSILON * magnitude;
    for (index, (got, want)) in rebuilt.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= bound,
            "blocked QR: {label} A[{index}] rebuilt as {got}, expected {want} (bound {bound})"
        );
    }
}

/// The identity's factor is upper-triangular and rebuilds the identity. The
/// reflector signs are backend-owned, so the oracle is the reconstruction, not
/// an exact **R**.
fn identity_factor_reconstructs_the_identity<B>(backend: &B, block_size: usize)
where
    B: BlockedQrBackend,
{
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let result = factor(backend, &identity, 2, 2, block_size).expect("identity factorization");

    assert_eq!(
        result.heads.len(),
        2,
        "blocked QR: one reflector per column"
    );
    assert_eq!(
        result.betas.len(),
        2,
        "blocked QR: one reflector per column"
    );
    assert_upper_triangular(backend, &result, 2, 2);
    assert_rebuilt(&reconstruct(&result, 2, 2), &identity, 1.0, 2, "identity");
}

/// A square fixture wider than one panel: the loop must scatter the first panel
/// and apply its reflectors to the trailing columns before the rest.
fn reconstructs_the_fixture_across_a_block_boundary<B>(backend: &B, block_size: usize)
where
    B: BlockedQrBackend,
{
    let n = block_size + 2;
    let host: Vec<f32> = (0..n * n)
        .map(|index| {
            let (row, col) = (index / n, index % n);
            if row == col {
                n as f32 + 4.0
            } else {
                0.1 / (1.0 + row.abs_diff(col) as f32)
            }
        })
        .collect();
    let magnitude = host.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));

    let result = factor(backend, &host, n, n, block_size).expect("fixture factorization");
    assert_eq!(
        result.heads.len(),
        n,
        "blocked QR: one reflector per column"
    );
    assert_upper_triangular(backend, &result, n, n);
    assert_rebuilt(
        &reconstruct(&result, n, n),
        &host,
        magnitude,
        n,
        "block-boundary",
    );
}

/// A tall rectangular fixture exercises a full final panel after an exact
/// panel boundary. The upper `n × n` block is the Leto differential; the
/// reconstruction above remains the backend-independent oracle for the packed
/// reflectors and the full `m × n` factor buffer.
fn rectangular_r_matches_the_leto_reference<B>(backend: &B, block_size: usize)
where
    B: BlockedQrBackend,
{
    let m = 2 * block_size + 6;
    let n = block_size + 3;
    let host: Vec<f32> = (0..m * n)
        .map(|index| {
            let (row, col) = (index / n, index % n);
            if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            }
        })
        .collect();
    let result = factor(backend, &host, m, n, block_size).expect("rectangular factorization");
    let actual = assert_upper_triangular(backend, &result, m, n);

    let matrix = leto::Array::from_shape_vec([m, n], host).expect("rectangular fixture");
    let reference = leto_ops::qr_decompose(&matrix.view()).expect("leto QR");
    let reference_r = reference.r();
    let expected = leto::Storage::as_slice(reference_r.storage());
    for row in 0..n {
        for col in 0..n {
            let index = row * n + col;
            let bound = 4.0 * m as f32 * f32::EPSILON * expected[index].abs().max(1.0);
            assert!(
                (actual[index] - expected[index]).abs() <= bound,
                "blocked QR: R({row},{col}) = {} vs leto {} (bound {bound})",
                actual[index],
                expected[index]
            );
        }
    }
}
