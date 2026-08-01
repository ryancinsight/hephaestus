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

    // R is upper-triangular. The buffer is either the n×n leading block or
    // the full m×n factor (per the seam contract); in both row-major shapes
    // every entry strictly below the diagonal of the first n rows — and any
    // row beyond n — vanishes to within factorization error.
    use hephaestus_core::DeviceBuffer;
    let r_len = qr.r_buffer().len();
    assert!(
        r_len == 4 || r_len == 6,
        "{name}: R buffer must be n*n or m*n, got {r_len}"
    );
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
