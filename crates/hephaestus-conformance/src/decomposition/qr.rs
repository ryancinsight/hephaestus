//! QR clauses: least squares, column pivoting, and their Leto differentials.

use hephaestus_core::{ComputeDevice, DecompositionOps, StridedView};
use leto::Layout;

use hephaestus_core::{ColPivQrHandle, QrHandle};

use super::tolerance::DECOMP_BOUND;

/// QR least squares on the consistent system `A·x = b` with
/// `A = [[1,0],[0,1],[1,1]]`, `b = [1,1,2]`: the minimizer is `x = [1,1]`
/// with zero residual, and the strict lower triangle of `R` vanishes.
pub(crate) fn qr_solves_a_consistent_least_squares_system<D, R>(device: &D, ops: &R)
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

/// QR least squares on the overdetermined consistent system
/// `A = [[1,0],[0,1],[1,1]]`, `b = [1,2,3]`: the exact minimizer is
/// `x = [1,2]`, asserted both analytically and against leto.
pub(crate) fn qr_matches_the_leto_reference<D, R>(device: &D, ops: &R)
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

/// Column-pivoted QR reveals rank and solves consistent least squares.
///
/// The rank-1 fixture stacks a column and its double, so exactly one
/// pivoted diagonal survives; the full-rank fixture reuses the QR
/// least-squares system whose minimizer is exactly `[1, 1]`.
pub(crate) fn col_piv_qr_reveals_rank_and_solves<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();

    let deficient = device
        .upload(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0])
        .expect("rank-1 upload");
    let tall = Layout::c_contiguous([3, 2]).expect("3x2 layout");
    let qr = ops
        .col_piv_qr(device, StridedView::new(&deficient, &tall))
        .expect("rank-1 col-pivoted QR");
    assert_eq!(qr.shape(), (3, 2), "{name}: QR shape");
    assert_eq!(qr.rank(), 1, "{name}: rank-1 fixture must reveal rank 1");
    let permutation = qr.permutation();
    assert_eq!(permutation.len(), 2, "{name}: permutation length");
    let mut seen = [false; 2];
    for &p in permutation {
        assert!(
            p < 2 && !seen[p],
            "{name}: invalid column permutation {permutation:?}"
        );
        seen[p] = true;
    }

    let full = device
        .upload(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("full-rank upload");
    let qr = ops
        .col_piv_qr(device, StridedView::new(&full, &tall))
        .expect("full-rank col-pivoted QR");
    assert_eq!(qr.rank(), 2, "{name}: full-rank fixture must reveal rank 2");
    let rhs = device.upload(&[1.0f32, 1.0, 2.0]).expect("rhs upload");
    let x = qr.solve_least_squares(device, &rhs).expect("LS solve");
    let mut got = [0.0f32; 2];
    device.download(&x, &mut got).expect("solution download");
    for (index, value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < DECOMP_BOUND,
            "{name}: pivoted LS component {index} = {value}, expected 1"
        );
    }
}

/// Column-pivoted QR of the rank-2 matrix `[[1,2,3],[4,5,6],[7,8,9]]`
/// agrees with leto on the revealed rank and the pivot choice (column
/// norms are distinct, so the choice is tie-free and deterministic).
pub(crate) fn col_piv_qr_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([3, 3]).expect("matrix layout");
    let decomp = ops
        .col_piv_qr(device, StridedView::new(&a, &layout))
        .expect("device column-pivoted QR");

    let host_matrix = leto::Array::from_shape_vec([3, 3], a_host.to_vec()).expect("host matrix");
    let host_decomp = leto_ops::col_piv_qr(&host_matrix.view()).expect("leto column-pivoted QR");

    assert_eq!(
        decomp.rank(),
        host_decomp.rank(),
        "{name}: column-pivoted QR rank diverges from leto"
    );
    assert_eq!(
        decomp.permutation(),
        host_decomp.permutation(),
        "{name}: column-pivoted QR pivot choice diverges from leto"
    );
}
