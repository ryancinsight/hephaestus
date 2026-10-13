//! Cholesky clauses: SPD factorization, indefinite rejection, Leto differential.

use hephaestus_core::{ComputeDevice, DecompositionOps, StridedView};
use leto::Layout;

use hephaestus_core::CholeskyHandle;

use super::common::DECOMP_BOUND;

/// Cholesky of the SPD matrix `[[4,2],[2,3]]`: `det = 8`,
/// `A·[1,1]ᵀ = [6,5]ᵀ`, and `L·Lᵀ` reconstructs `A`.
pub(crate) fn cholesky_factors_an_spd_matrix<D, R>(device: &D, ops: &R)
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
pub(crate) fn cholesky_rejects_an_indefinite_matrix<D, R>(device: &D, ops: &R)
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

/// Cholesky of the 3x3 SPD matrix `[[4,2,0.5],[2,5,1],[0.5,1,3]]` matches
/// leto elementwise on the lower factor within the derived bound.
pub(crate) fn cholesky_matches_the_leto_reference<D, R>(device: &D, ops: &R)
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
