//! LU clauses: partial pivoting, full pivoting, and their Leto differentials.

use hephaestus_core::{ComputeDevice, DecompositionOps, StridedView};
use leto::Layout;

use hephaestus_core::{FullPivLuHandle, LuHandle};

use super::tolerance::DECOMP_BOUND;

/// LU with a forced pivot: `A = [[0,2,1],[1,1,1],[2,4,1]]` has `a₀₀ = 0`,
/// `det A = 4`, and `A·[1,1,1]ᵀ = [3,3,7]ᵀ`.
pub(crate) fn lu_pivots_solves_and_reconstructs<D, R>(device: &D, ops: &R)
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
pub(crate) fn lu_rejects_a_non_square_operand<D, R>(device: &D, ops: &R)
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

/// Differential LU clause against leto, the drop-in CPU substrate.
///
/// Device and host run the same algorithm family in `f32` but not in a
/// provably identical evaluation order, so factor equality is
/// epsilon-bounded by the module-head derivation, never bitwise
/// (reduction-order sensitivity). Pivot choices ARE asserted exactly: the
/// fixture has a unique maximum-magnitude candidate at every elimination
/// step, so any correct partial-pivoting implementation selects the same
/// rows.
pub(crate) fn lu_matches_the_leto_reference<D, R>(device: &D, ops: &R)
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

/// Fully pivoted LU reveals rank, keeps exact det/solve oracles, and
/// reconstructs `P·A·Q = L·U` through both permutation vectors in `f64`.
pub(crate) fn full_piv_lu_reveals_rank_and_reconstructs<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();

    // Rank-deficient: row 2 is twice row 1.
    let deficient = device
        .upload(&[1.0f32, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0])
        .expect("rank-2 upload");
    let square = Layout::c_contiguous([3, 3]).expect("3x3 layout");
    let lu = ops
        .full_piv_lu(device, StridedView::new(&deficient, &square))
        .expect("rank-2 full-pivoted LU");
    assert_eq!(lu.rank(), 2, "{name}: rank-2 fixture must reveal rank 2");

    // Full-rank: the trio fixture with det 4 and solution [1,1,1].
    let a_host = [0.0f32, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let lu = ops
        .full_piv_lu(device, StridedView::new(&a, &square))
        .expect("full-rank full-pivoted LU");
    assert_eq!(lu.order(), 3, "{name}: order");
    assert_eq!(lu.rank(), 3, "{name}: full-rank fixture must reveal rank 3");
    assert!(
        (lu.det() - 4.0).abs() < DECOMP_BOUND,
        "{name}: full-pivot det {} != 4",
        lu.det()
    );
    let rhs = device.upload(&[3.0f32, 3.0, 7.0]).expect("rhs upload");
    let x = lu.solve(device, &rhs).expect("full-pivot solve");
    let mut got = [0.0f32; 3];
    device.download(&x, &mut got).expect("solution download");
    for (index, value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < DECOMP_BOUND,
            "{name}: full-pivot solution component {index} = {value}, expected 1"
        );
    }

    // Both permutation vectors are permutations.
    let rows = lu.row_permutation();
    let cols = lu.col_permutation();
    for (label, permutation) in [("row", rows), ("column", cols)] {
        assert_eq!(permutation.len(), 3, "{name}: {label} permutation length");
        let mut seen = [false; 3];
        for &p in permutation {
            assert!(
                p < 3 && !seen[p],
                "{name}: invalid {label} permutation {permutation:?}"
            );
            seen[p] = true;
        }
    }

    // Host reconstruction in f64: gather A through both permutations and
    // compare against L·U from the packed factors.
    let mut factors = [0.0f32; 9];
    device
        .download(lu.factors(), &mut factors)
        .expect("factor download");
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
            let gathered = f64::from(a_host[rows[row] * 3 + cols[col]]);
            max_err = max_err.max((acc - gathered).abs());
        }
    }
    assert!(
        max_err < f64::from(DECOMP_BOUND),
        "{name}: full-pivot reconstruction error {max_err}"
    );
}

/// Fully pivoted LU of `[[1,2],[3,4]]` agrees with leto on rank, both
/// permutations, the packed factors, the determinant, and the solution
/// of `A·x = [5,11]ᵀ` (which is `x = [1,2]ᵀ`).
pub(crate) fn full_piv_lu_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([2, 2]).expect("matrix layout");
    let decomp = ops
        .full_piv_lu(device, StridedView::new(&a, &layout))
        .expect("device fully pivoted LU");

    let host_matrix = leto::Array::from_shape_vec([2, 2], a_host.to_vec()).expect("host matrix");
    let host_decomp = leto_ops::full_piv_lu(&host_matrix.view()).expect("leto fully pivoted LU");

    assert_eq!(decomp.order(), 2, "{name}: fully pivoted LU order");
    assert_eq!(
        decomp.rank(),
        host_decomp.rank(),
        "{name}: fully pivoted LU rank diverges from leto"
    );
    assert_eq!(
        decomp.row_permutation(),
        host_decomp.row_permutation(),
        "{name}: fully pivoted LU row pivots diverge from leto"
    );
    assert_eq!(
        decomp.col_permutation(),
        host_decomp.col_permutation(),
        "{name}: fully pivoted LU column pivots diverge from leto"
    );
    assert!(
        (decomp.det() - host_decomp.det()).abs() < DECOMP_BOUND,
        "{name}: fully pivoted LU det {} vs leto {}",
        decomp.det(),
        host_decomp.det()
    );

    let mut factors = [0.0f32; 4];
    device
        .download(decomp.factors(), &mut factors)
        .expect("factor download");
    for (index, (device_value, host_value)) in
        factors.iter().zip(host_decomp.lu_factors()).enumerate()
    {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: fully pivoted LU factor {index}: device {device_value} vs leto {host_value}"
        );
    }

    let rhs_host = [5.0f32, 11.0];
    let rhs = device.upload(&rhs_host).expect("rhs upload");
    let solution = decomp.solve(device, &rhs).expect("device solve");
    let mut got = [0.0f32; 2];
    device
        .download(&solution, &mut got)
        .expect("solution download");
    let host_rhs = leto::Array::from_shape_vec([2], rhs_host.to_vec()).expect("host rhs");
    let host_solution = host_decomp.solve(&host_rhs.view()).expect("leto solve");
    for (index, (device_value, host_value)) in got
        .iter()
        .zip(leto::Storage::as_slice(host_solution.storage()))
        .enumerate()
    {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: fully pivoted LU solution {index}: device {device_value} vs leto {host_value}"
        );
    }
}
