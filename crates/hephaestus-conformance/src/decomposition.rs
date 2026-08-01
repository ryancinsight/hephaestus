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
//! `n ≤ 3` and infinity-norm condition numbers below 32 by construction,
//! so `c(n)·ε·κ ≤ 3²·1.2e-7·32 ≈ 3.5e-4`; the clauses assert `1e-3`,
//! roughly 3× that bound, and reconstruction runs in `f64` on the host so
//! the comparison itself adds no `f32` error.

use hephaestus_core::{
    CholeskyHandle, ComputeDevice, DecompositionOps, LuHandle, QrHandle, StridedView,
};
use leto::Layout;

/// Derived clause bound; see the module-level derivation.
const DECOMP_BOUND: f32 = 1e-3;

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
    lu_pivots_solves_and_reconstructs(device, ops);
    lu_rejects_a_non_square_operand(device, ops);
    cholesky_factors_an_spd_matrix(device, ops);
    cholesky_rejects_an_indefinite_matrix(device, ops);
    qr_solves_a_consistent_least_squares_system(device, ops);
    lu_matches_the_leto_reference(device, ops);
    cholesky_matches_the_leto_reference(device, ops);
    qr_matches_the_leto_reference(device, ops);
    identity_factorizations_are_exact(device, ops);
}

/// LU with a forced pivot: `A = [[0,2,1],[1,1,1],[2,4,1]]` has `a₀₀ = 0`,
/// `det A = 4`, and `A·[1,1,1]ᵀ = [3,3,7]ᵀ`.
fn lu_pivots_solves_and_reconstructs<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [0.0f32, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([3, 3]).expect("matrix layout");
    let lu = ops
        .lu(device, StridedView::new(&a, &layout))
        .expect("LU with a zero leading pivot must succeed via pivoting");

    assert_eq!(lu.order(), 3, "{name}: LU order");
    assert!(
        (lu.det() - 4.0).abs() < DECOMP_BOUND,
        "{name}: det {} != 4",
        lu.det()
    );

    // The pivot indices must form a valid permutation domain.
    let pivots = lu.pivots();
    assert_eq!(pivots.len(), 3, "{name}: pivot count");
    assert!(
        pivots.iter().all(|&p| p < 3),
        "{name}: pivot indices in range, got {pivots:?}"
    );

    // Solve A·x = [3,3,7] and recover x = [1,1,1].
    let rhs = device.upload(&[3.0f32, 3.0, 7.0]).expect("rhs upload");
    let x = lu.solve(device, &rhs).expect("LU solve");
    let mut got = [0.0f32; 3];
    device.download(&x, &mut got).expect("solution download");
    for (index, value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < DECOMP_BOUND,
            "{name}: LU solution component {index} = {value}, expected 1"
        );
    }

    // The pivot vector is a permutation: each source row appears exactly
    // once (leto's convention — row `k` of `P·A` is row `pivots[k]` of `A`).
    let mut seen = [false; 3];
    for &p in pivots {
        assert!(!seen[p], "{name}: duplicate pivot index {p}");
        seen[p] = true;
    }

    // Host-side reconstruction in f64: P·A (rows gathered through the pivot
    // vector) must equal L·U from the packed factors.
    let mut factors = [0.0f32; 9];
    device
        .download(lu.factors(), &mut factors)
        .expect("factor download");
    let mut permuted = [0.0f64; 9];
    for (k, &src) in pivots.iter().enumerate() {
        for col in 0..3 {
            permuted[k * 3 + col] = f64::from(a_host[src * 3 + col]);
        }
    }
    let mut max_err = 0.0f64;
    for row in 0..3 {
        for col in 0..3 {
            let mut acc = 0.0f64;
            for k in 0..3 {
                let l = match k.cmp(&row) {
                    std::cmp::Ordering::Less => f64::from(factors[row * 3 + k]),
                    std::cmp::Ordering::Equal => 1.0,
                    std::cmp::Ordering::Greater => 0.0,
                };
                let u = if k <= col {
                    f64::from(factors[k * 3 + col])
                } else {
                    0.0
                };
                acc += l * u;
            }
            max_err = max_err.max((acc - permuted[row * 3 + col]).abs());
        }
    }
    assert!(
        max_err < f64::from(DECOMP_BOUND),
        "{name}: LU reconstruction error {max_err}"
    );
}

/// A non-square operand is rejected as a typed error.
fn lu_rejects_a_non_square_operand<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a = device
        .upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("matrix upload");
    let layout = Layout::c_contiguous([2, 3]).expect("matrix layout");
    assert!(
        ops.lu(device, StridedView::new(&a, &layout)).is_err(),
        "{name}: LU must reject a non-square operand"
    );
}

/// Cholesky of the SPD matrix `[[4,2],[2,3]]`: `det = 8`,
/// `A·[1,1]ᵀ = [6,5]ᵀ`, and `L·Lᵀ` reconstructs `A`.
fn cholesky_factors_an_spd_matrix<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a = device.upload(&[4.0f32, 2.0, 2.0, 3.0]).expect("upload");
    let layout = Layout::c_contiguous([2, 2]).expect("layout");
    let chol = ops
        .cholesky(device, StridedView::new(&a, &layout))
        .expect("SPD Cholesky");

    assert_eq!(chol.order(), 2, "{name}: Cholesky order");
    assert!(
        (chol.det() - 8.0).abs() < DECOMP_BOUND,
        "{name}: Cholesky det {} != 8",
        chol.det()
    );

    let rhs = device.upload(&[6.0f32, 5.0]).expect("rhs upload");
    let x = chol.solve(device, &rhs).expect("Cholesky solve");
    let mut got = [0.0f32; 2];
    device.download(&x, &mut got).expect("solution download");
    for (index, value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < DECOMP_BOUND,
            "{name}: Cholesky solution component {index} = {value}, expected 1"
        );
    }

    // Reconstruct A = L·Lᵀ in f64 from the lower triangle only.
    let mut lower = [0.0f32; 4];
    device
        .download(chol.lower(), &mut lower)
        .expect("factor download");
    let l = |row: usize, col: usize| -> f64 {
        if col <= row {
            f64::from(lower[row * 2 + col])
        } else {
            0.0
        }
    };
    let expected = [4.0f64, 2.0, 2.0, 3.0];
    for row in 0..2 {
        for col in 0..2 {
            let acc = l(row, 0) * l(col, 0) + l(row, 1) * l(col, 1);
            assert!(
                (acc - expected[row * 2 + col]).abs() < f64::from(DECOMP_BOUND),
                "{name}: Cholesky reconstruction ({row},{col}) = {acc}"
            );
        }
    }
}

/// An indefinite symmetric matrix has no real Cholesky factorization.
fn cholesky_rejects_an_indefinite_matrix<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a = device.upload(&[1.0f32, 2.0, 2.0, 1.0]).expect("upload");
    let layout = Layout::c_contiguous([2, 2]).expect("layout");
    assert!(
        ops.cholesky(device, StridedView::new(&a, &layout)).is_err(),
        "{name}: Cholesky must reject an indefinite matrix"
    );
}

/// QR least squares on the consistent system `A·x = b` with
/// `A = [[1,0],[0,1],[1,1]]`, `b = [1,1,2]`: the minimizer is `x = [1,1]`
/// with zero residual, and the strict lower triangle of `R` vanishes.
fn qr_solves_a_consistent_least_squares_system<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a = device
        .upload(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("upload");
    let layout = Layout::c_contiguous([3, 2]).expect("layout");
    let qr = ops
        .qr(device, StridedView::new(&a, &layout))
        .expect("QR of a full-rank tall matrix");

    assert_eq!(qr.shape(), (3, 2), "{name}: QR shape");

    let rhs = device.upload(&[1.0f32, 1.0, 2.0]).expect("rhs upload");
    let x = qr.solve_least_squares(device, &rhs).expect("QR solve");
    let mut got = [0.0f32; 2];
    device.download(&x, &mut got).expect("solution download");
    for (index, value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < DECOMP_BOUND,
            "{name}: QR solution component {index} = {value}, expected 1"
        );
    }

    // R is m×n row-major (the seam pins leto's convention): every entry
    // strictly below the diagonal — including the rows beyond n — vanishes
    // to within factorization error.
    use hephaestus_core::DeviceBuffer;
    let r_len = qr.r_buffer().len();
    assert_eq!(r_len, 6, "{name}: R must be m*n = 6 elements, got {r_len}");
    let mut r = vec![0.0f32; r_len];
    device.download(qr.r_buffer(), &mut r).expect("R download");
    for (index, value) in r.iter().enumerate() {
        let (row, col) = (index / 2, index % 2);
        if col < row {
            assert!(
                value.abs() < DECOMP_BOUND,
                "{name}: R entry ({row},{col}) = {value} must vanish"
            );
        }
    }
}

/// Differential LU clause against leto, the drop-in CPU substrate.
///
/// Device and host run the same algorithm family in `f32` but not in a
/// provably identical evaluation order, so factor equality is
/// epsilon-bounded by the module-head derivation, never bitwise
/// (reduction-order sensitivity). Pivot choices ARE asserted exactly: the
/// fixture has a unique maximum-magnitude candidate at every elimination
/// step, so any correct partial-pivoting implementation selects the same
/// rows.
fn lu_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [0.0f32, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([3, 3]).expect("matrix layout");
    let lu = ops
        .lu(device, StridedView::new(&a, &layout))
        .expect("device LU");

    let host_matrix = leto::Array::from_shape_vec([3, 3], a_host.to_vec()).expect("host matrix");
    let host_lu = leto_ops::lu_decompose(&host_matrix.view()).expect("leto LU");

    assert_eq!(
        lu.pivots(),
        host_lu.pivots(),
        "{name}: LU pivot choice diverges from leto"
    );
    let mut factors = [0.0f32; 9];
    device
        .download(lu.factors(), &mut factors)
        .expect("factor download");
    let host_factors = leto::Storage::as_slice(host_lu.factors().storage());
    for (index, (device_value, host_value)) in factors.iter().zip(host_factors).enumerate() {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: LU factor {index}: device {device_value} vs leto {host_value}"
        );
    }
    assert!(
        (lu.det() - host_lu.det()).abs() < DECOMP_BOUND,
        "{name}: LU det {} vs leto {}",
        lu.det(),
        host_lu.det()
    );
}

/// Cholesky of the 3x3 SPD matrix `[[4,2,0.5],[2,5,1],[0.5,1,3]]` matches
/// leto elementwise on the lower factor within the derived bound.
fn cholesky_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [4.0f32, 2.0, 0.5, 2.0, 5.0, 1.0, 0.5, 1.0, 3.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([3, 3]).expect("matrix layout");
    let chol = ops
        .cholesky(device, StridedView::new(&a, &layout))
        .expect("device Cholesky");

    let host_matrix = leto::Array::from_shape_vec([3, 3], a_host.to_vec()).expect("host matrix");
    let host_chol = leto_ops::cholesky_decompose(&host_matrix.view()).expect("leto Cholesky");

    let mut lower = [0.0f32; 9];
    device
        .download(chol.lower(), &mut lower)
        .expect("factor download");
    let host_lower = leto::Storage::as_slice(host_chol.lower().storage());
    for row in 0..3 {
        for col in 0..=row {
            let index = row * 3 + col;
            assert!(
                (lower[index] - host_lower[index]).abs() < DECOMP_BOUND,
                "{name}: Cholesky lower ({row},{col}): device {} vs leto {}",
                lower[index],
                host_lower[index]
            );
        }
    }
}

/// QR least squares on the overdetermined consistent system
/// `A = [[1,0],[0,1],[1,1]]`, `b = [1,2,3]`: the exact minimizer is
/// `x = [1,2]`, asserted both analytically and against leto.
fn qr_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([3, 2]).expect("matrix layout");
    let qr = ops
        .qr(device, StridedView::new(&a, &layout))
        .expect("device QR");

    let rhs_host = [1.0f32, 2.0, 3.0];
    let rhs = device.upload(&rhs_host).expect("rhs upload");
    let x = qr.solve_least_squares(device, &rhs).expect("device solve");
    let mut got = [0.0f32; 2];
    device.download(&x, &mut got).expect("solution download");

    let host_matrix = leto::Array::from_shape_vec([3, 2], a_host.to_vec()).expect("host matrix");
    let host_rhs = leto::Array::from_shape_vec([3], rhs_host.to_vec()).expect("host rhs");
    let host_qr = leto_ops::qr_decompose(&host_matrix.view()).expect("leto QR");
    let host_x = host_qr
        .solve_least_squares(&host_rhs.view())
        .expect("leto solve");
    let host_solution = leto::Storage::as_slice(host_x.storage());

    let expected = [1.0f32, 2.0];
    for index in 0..2 {
        assert!(
            (got[index] - expected[index]).abs() < DECOMP_BOUND,
            "{name}: QR minimizer component {index} = {}, expected {}",
            got[index],
            expected[index]
        );
        assert!(
            (got[index] - host_solution[index]).abs() < DECOMP_BOUND,
            "{name}: QR solution component {index}: device {} vs leto {}",
            got[index],
            host_solution[index]
        );
    }
}

/// Factorizing the identity is exact: every arithmetic step multiplies by
/// 0 or 1 or takes `sqrt(1)`, so no rounding occurs and the factors admit
/// exact equality (LU packed factors and pivots, Cholesky lower). QR
/// asserts entry magnitudes within the derived bound only: the Householder
/// sign convention belongs to the backend and is not contract.
fn identity_factorizations_are_exact<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let layout = Layout::c_contiguous([2, 2]).expect("layout");

    let a = device.upload(&identity).expect("upload");
    let lu = ops
        .lu(device, StridedView::new(&a, &layout))
        .expect("identity LU");
    let mut factors = [0.0f32; 4];
    device
        .download(lu.factors(), &mut factors)
        .expect("factor download");
    assert_eq!(factors, identity, "{name}: identity LU factors");
    assert_eq!(lu.pivots(), &[0, 1], "{name}: identity LU pivots");
    assert!(
        (lu.det() - 1.0).abs() == 0.0,
        "{name}: identity LU det {}",
        lu.det()
    );

    let a = device.upload(&identity).expect("upload");
    let chol = ops
        .cholesky(device, StridedView::new(&a, &layout))
        .expect("identity Cholesky");
    let mut lower = [0.0f32; 4];
    device
        .download(chol.lower(), &mut lower)
        .expect("factor download");
    assert_eq!(lower, identity, "{name}: identity Cholesky lower");

    let a = device.upload(&identity).expect("upload");
    let qr = ops
        .qr(device, StridedView::new(&a, &layout))
        .expect("identity QR");
    use hephaestus_core::DeviceBuffer;
    let mut r = vec![0.0f32; qr.r_buffer().len()];
    device.download(qr.r_buffer(), &mut r).expect("R download");
    for (index, value) in r.iter().take(4).enumerate() {
        let expected = identity[index];
        assert!(
            (value.abs() - expected).abs() < DECOMP_BOUND,
            "{name}: identity QR R entry {index} = {value}, expected magnitude {expected}"
        );
    }
}
