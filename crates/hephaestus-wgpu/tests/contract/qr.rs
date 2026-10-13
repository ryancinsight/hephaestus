use super::*;

/// The blocked-QR differential is one shared clause (SUBSTRATE-003): the
/// fixtures and assertions live in `hephaestus-conformance`, generic over the
/// `BlockedQrBackend` seam the shared `blocked_qr` loop drives. This
/// instantiation supplies the device and the panel width the blocked entry
/// point factors with.
#[cfg(feature = "decomposition")]
pub(crate) fn blocked_qr_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    const BLOCK_SIZE: usize = 32;
    hephaestus_conformance::assert_blocked_qr_contract(&device, BLOCK_SIZE);
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_qr_least_squares_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, qr_decompose_blocked};
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
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([m, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([m, n], matrix_host.clone()).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();
    let gpu_qr = qr_decompose_blocked(
        &device,
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
    let rhs = device.upload(&rhs_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([m], rhs_host).unwrap();
    let solution = gpu_qr.solve_least_squares(&device, &rhs).unwrap();
    let expected_solution = leto_qr.solve_least_squares(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; n];
    device.download(&solution, &mut got).unwrap();
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

/// `r_buffer()` denotes the same mathematical object on both QR entry points,
/// in both of the blocked entry's routing regimes.
///
/// `qr_decompose` uploads the host factor's **R**. `qr_decompose_blocked`
/// delegates to it at or below `QR_DIRECT_PANEL_LIMIT` (4) panels of
/// `QR_BLOCK_SIZE` (32) columns, or below `QR_DIRECT_ROW_LIMIT` (256) rows,
/// and only past both runs the device schedule whose panel write-back zeroes
/// every `col < row` entry before the buffer reaches the caller. Both regimes
/// are exercised here: a shape at `n = 35` (2 panels, delegating) and one at
/// `256x129` (5 panels and 256 rows, device path). A shape inside the
/// delegating regime alone would assert nothing about the device write-back —
/// the two handles would be the same uploaded buffer.
#[cfg(feature = "decomposition")]
pub(crate) fn qr_r_buffer_is_upper_triangular_on_both_entry_points() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // (m, n, exercises the device schedule)
    for (m, n, blocked_route) in [(70usize, 35usize, false), (256usize, 129usize, true)] {
        qr_r_buffer_contract_at_shape(&device, m, n, blocked_route);
    }
}

/// The device-accumulated **Q** must match leto's host accumulation at both
/// QR routing regimes, and must be orthogonal.
#[cfg(feature = "decomposition")]
pub(crate) fn qr_accumulated_q_matches_host_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // (m, n, exercises the device schedule, checks orthogonality)
    for (m, n, blocked_route, check_orthogonality) in [
        (70usize, 35usize, false, false),
        (256usize, 129usize, true, true),
    ] {
        qr_accumulated_q_contract_at_shape(&device, m, n, blocked_route, check_orthogonality);
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_qr_preserves_panel_boundary_contracts() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    // Tall enough for the 129-column case to exercise the device schedule
    // under the row rule; narrower cases still delegate via the panel rule.
    for cols in [32usize, 33, 35, 64, 65, 128, 129] {
        let rows = 256usize;
        let mut matrix_host = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                matrix_host[row * cols + col] = if row == col {
                    6.0
                } else {
                    0.02 / (1.0 + row.abs_diff(col) as f32)
                };
            }
        }
        let matrix = device.upload(&matrix_host).unwrap();
        let layout = Layout::c_contiguous([rows, cols]).unwrap();
        let leto_matrix = leto::Array::from_shape_vec([rows, cols], matrix_host.clone()).unwrap();
        let expected_decomposition = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();
        let expected = expected_decomposition.r();
        let expected = leto::Storage::as_slice(expected.storage());

        let qr = qr_decompose_blocked(
            &device,
            StridedOperand {
                buffer: &matrix,
                layout: &layout,
            },
        )
        .unwrap();
        let mut actual = vec![0.0f32; rows * cols];
        device.download(qr.r_buffer(), &mut actual).unwrap();

        // Each output accumulates at most `cols` reflector-rounding stages.
        // The factor 8 covers one multiply-add pair plus norm/scale rounding per
        // stage, giving the standard O(n·ε) backward-error envelope.
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 8.0 * cols as f32 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{rows}x{cols} blocked QR R[{index}] = {actual}, expected {expected}, \
                tolerance {tolerance}"
            );
        }

        let expected_solution: Vec<f32> = (0..cols)
            .map(|col| 0.5 + col as f32 / cols as f32)
            .collect();
        let rhs_host: Vec<f32> = matrix_host
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(&expected_solution)
                    .map(|(&coefficient, &value)| coefficient * value)
                    .sum()
            })
            .collect();
        let rhs = device.upload(&rhs_host).unwrap();
        let solution = qr.solve_least_squares(&device, &rhs).unwrap();
        let mut actual_solution = vec![0.0f32; cols];
        device.download(&solution, &mut actual_solution).unwrap();

        // Strict diagonal dominance bounds the infinity-norm condition number
        // near one for this matrix family. The factor 64 covers QR and
        // triangular-solve rounding over `cols` stages.
        let solve_tolerance = 64.0 * cols as f32 * f32::EPSILON;
        for (col, (&actual, &expected)) in
            actual_solution.iter().zip(&expected_solution).enumerate()
        {
            assert!(
                (actual - expected).abs() <= solve_tolerance,
                "{rows}x{cols} blocked QR solve x[{col}] = {actual}, expected {expected}, \
                 tolerance {solve_tolerance}"
            );
        }
        for (row, coefficients) in matrix_host.chunks_exact(cols).enumerate() {
            let reconstructed: f32 = coefficients
                .iter()
                .zip(&actual_solution)
                .map(|(&coefficient, &value)| coefficient * value)
                .sum();
            let tolerance = 64.0 * cols as f32 * f32::EPSILON * rhs_host[row].abs().max(1.0);
            assert!(
                (reconstructed - rhs_host[row]).abs() <= tolerance,
                "{rows}x{cols} blocked QR reconstructed rhs[{row}] = {reconstructed}, \
                 expected {}, tolerance {tolerance}",
                rhs_host[row]
            );
        }
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_qr_identity_yields_identity_r() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (2, 2));

    let mut got_r = vec![0.0f32; 4];
    device.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
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
pub(crate) fn blocked_qr_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]  =>  x = [1, 2]
    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_qr.solve_least_squares(&device, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();

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
pub(crate) fn blocked_qr_rejects_underdetermined() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    let host = vec![0.0f32; 6];
    let input = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = qr_decompose_blocked(
        &device,
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
pub(crate) fn blocked_qr_rejects_non_dense_operands() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_blocked_rejects_non_dense(&device, hephaestus_wgpu::qr_decompose_blocked, "QR");
}

#[cfg(feature = "decomposition")]
pub(crate) fn empty_qr_preserves_shape_and_identity() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let empty = device.alloc_zeroed::<f32>(0).unwrap();
    let layout = leto::Layout::c_contiguous([3, 0]).unwrap();
    let operand = hephaestus_wgpu::StridedOperand {
        buffer: &empty,
        layout: &layout,
    };
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    for qr in [
        hephaestus_wgpu::qr_decompose(&device, operand).unwrap(),
        hephaestus_wgpu::qr_decompose_blocked(&device, operand).unwrap(),
    ] {
        assert_eq!(qr.shape(), (3, 0));
        assert_eq!(qr.r_buffer().len(), 0);
        assert_eq!(qr.inner().shape(), (3, 0));
        assert_eq!(leto::Storage::as_slice(qr.inner().q().storage()), identity);
        assert_eq!(qr.inner().r().shape(), [3, 0]);
    }
}
