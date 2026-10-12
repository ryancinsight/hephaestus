use super::*;

/// The blocked-QR differential is one shared clause (SUBSTRATE-003): the
/// fixtures and assertions live in `hephaestus-conformance`, generic over the
/// `BlockedQrBackend` seam the shared `blocked_qr` loop drives.
#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_contract() {
    let Some(dev) = device("blocked_qr_contract") else {
        return;
    };
    const BLOCK_SIZE: usize = 32;
    hephaestus_conformance::assert_blocked_qr_contract(&dev, BLOCK_SIZE);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_least_squares_matches_leto_reference() {
    let Some(dev) = device("blocked_qr_least_squares_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    // 70×35 matrix exercises two QR blocks (QR_BLOCK_SIZE = 32).
    let (m, n) = (70, 35);
    let mut matrix_host = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            matrix_host[row * n + col] = if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([m, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([m, n], matrix_host.clone()).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();
    let gpu_qr = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    // The shared blocked-QR clause owns the R differential. This wrapper-level
    // clause retains only the backend solve path, which the core loop does not
    // expose.
    let rhs_host: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
    let rhs = dev.upload(&rhs_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([m], rhs_host).unwrap();
    let solution = gpu_qr.solve_least_squares(&dev, &rhs).unwrap();
    let expected_solution = leto_qr.solve_least_squares(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; n];
    dev.download(&solution, &mut got).unwrap();
    let expected = leto::Storage::as_slice(expected_solution.storage());
    // For this near-consistent, well-conditioned system (κ₂ ≈ 1, so the
    // κ²·residual term of least-squares sensitivity is negligible) two
    // backward-stable solves differ by ≤ 2·c(m)·ε·‖x‖∞ with c(m) ≤ 4·m;
    // 8·m·ε·‖x‖∞ doubles that.
    let x_inf = expected.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
    let solve_bound = 8.0 * m as f32 * f32::EPSILON * x_inf;
    for i in 0..n {
        assert!(
            (got[i] - expected[i]).abs() <= solve_bound,
            "blocked QR solve x[{i}] = {} expected {}",
            got[i],
            expected[i]
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_identity_yields_identity_r() {
    let Some(dev) = device("blocked_qr_identity_yields_identity_r") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (2, 2));

    let mut got_r = vec![0.0f32; 4];
    dev.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
    let r_ref = leto_qr.r();
    let expected_r = leto::Storage::as_slice(r_ref.storage());
    for i in 0..4 {
        // Identity input reaches R through at most one Householder
        // reflection over exact dyadic values: ≤ 4 roundings per entry.
        let tolerance = 8.0 * f32::EPSILON * expected_r[i].abs().max(1.0);
        assert!(
            (got_r[i] - expected_r[i]).abs() <= tolerance,
            "blocked QR R[{i}] = {} expected {}",
            got_r[i],
            expected_r[i]
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_solve_known_system_accurate() {
    let Some(dev) = device("blocked_qr_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]  =>  x = [1, 2]
    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_qr.solve_least_squares(&dev, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();

    // The system is consistent with exact solution [1, 2]; Householder
    // QR at m = 3 spends tens of flops per entry, so the residual is
    // bounded by c·ε·‖b‖∞ with c ≲ 32.
    let residual_bound = 32.0 * f32::EPSILON * 3.0;
    let residual_0 = 1.0 * got[0] + 0.0 * got[1] - 1.0;
    let residual_1 = 0.0 * got[0] + 1.0 * got[1] - 2.0;
    let residual_2 = 1.0 * got[0] + 1.0 * got[1] - 3.0;
    assert!(
        residual_0.abs() <= residual_bound,
        "blocked QR residual[0] = {residual_0}"
    );
    assert!(
        residual_1.abs() <= residual_bound,
        "blocked QR residual[1] = {residual_1}"
    );
    assert!(
        residual_2.abs() <= residual_bound,
        "blocked QR residual[2] = {residual_2}"
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_rejects_underdetermined() {
    let Some(dev) = device("blocked_qr_rejects_underdetermined") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    let host = vec![0.0f32; 6];
    let input = dev.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
    );
    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message }) if message.contains("m ≥ n")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_rejects_non_dense_operands() {
    let Some(dev) = device("blocked_qr_rejects_non_dense_operands") else {
        return;
    };
    use hephaestus_cuda::qr_decompose_blocked;
    assert_blocked_rejects_non_dense(&dev, qr_decompose_blocked, "QR");
}
