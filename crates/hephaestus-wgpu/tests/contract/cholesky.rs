use super::*;

#[cfg(feature = "decomposition")]
/// The blocked-Cholesky differential is one shared clause (SUBSTRATE-003):
/// the fixtures and assertions live in `hephaestus-conformance`, generic over
/// the `BlockedCholeskyBackend` seam the shared `blocked_cholesky` loop drives.
/// This instantiation supplies the device and the panel width the blocked
/// entry point factors with.
#[cfg(feature = "decomposition")]
pub(crate) fn blocked_cholesky_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    const BLOCK_SIZE: usize = 64;
    hephaestus_conformance::assert_blocked_cholesky_contract(&device, BLOCK_SIZE);
}

/// The device factor's strictly-upper triangle is exactly zero, including the
/// region outside the diagonal blocks.
///
/// The blocked path's per-panel scatters cover only `row >= blockstart(col)`,
/// and the panel factorisation zeroes just the diagonal blocks' own upper
/// triangle, so cells strictly above the diagonal *outside* those blocks hold
/// the input's values until the closing triangular-zero pass clears them.
/// This fixture therefore carries a large, distinctive off-diagonal value: a
/// regression in that pass leaves `0.5` in the factor rather than something
/// that could be mistaken for rounding noise. `n` exceeds the 64-element
/// block size so at least one such region exists.
#[cfg(feature = "decomposition")]
pub(crate) fn blocked_cholesky_zeroes_strict_upper_outside_diagonal_blocks() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, cholesky_decompose_blocked};
    use leto::Layout;

    let n = 96usize;
    // A = (n - 0.5)·I + 0.5·J is symmetric positive-definite (eigenvalues
    // n - 0.5 and 1.5·n - 0.5) and strongly diagonally dominant.
    let mut matrix_host = vec![0.5f32; n * n];
    for index in 0..n {
        matrix_host[index * n + index] = n as f32;
    }

    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_cholesky = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got_lower = vec![0.0f32; n * n];
    device
        .download(gpu_cholesky.lower(), &mut got_lower)
        .unwrap();

    for row in 0..n {
        for col in (row + 1)..n {
            let value = got_lower[row * n + col];
            assert_eq!(
                value, 0.0,
                "strict upper entry [{row}, {col}] must be zero, got {value}"
            );
        }
    }
    // A wholly zeroed buffer would satisfy the loop above; the diagonal pins
    // that the factor itself survived the pass.
    for index in 0..n {
        let diagonal = got_lower[index * n + index];
        assert!(
            diagonal > 0.0,
            "diagonal entry [{index}, {index}] must stay positive, got {diagonal}"
        );
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_cholesky_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, cholesky_decompose_blocked};
    use leto::Layout;

    // A = [[4, 2], [2, 3]], b = [8, 7]  =>  x = [1.25, 1.5]
    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![8.0f32, 7.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_chol.solve(&device, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();
    assert!(
        (got[0] - 1.25f32).abs() <= 1e-5,
        "x[0] = {} expected 1.25",
        got[0]
    );
    assert!(
        (got[1] - 1.5f32).abs() <= 1e-5,
        "x[1] = {} expected 1.5",
        got[1]
    );

    // Verify residual A*x ≈ b.
    let ax0 = 4.0 * got[0] + 2.0 * got[1];
    let ax1 = 2.0 * got[0] + 3.0 * got[1];
    assert!((ax0 - 8.0).abs() <= 1e-4, "residual[0] = {ax0}");
    assert!((ax1 - 7.0).abs() <= 1e-4, "residual[1] = {ax1}");
}

/// Device forward/backward substitution across two solve blocks (`n = 300`:
/// one full 256-row block plus a ragged 44-row tail) matches leto's host
/// substitution against the same downloaded factor, and never materializes
/// the host factor.
///
/// Fixture: `A = L·Lᵀ` with `L = I − P`, `P ≥ 0` strictly lower and every row
/// and column sum of `P` at most `1/2` (`|P[i,j]| ≤ 1/(2n)`). Then
/// `‖L‖∞, ‖L‖₁ ≤ 3/2` and, by the Neumann series, `‖L⁻¹‖∞, ‖L⁻¹‖₁ ≤ 2`, so
/// the recurrence is well conditioned: `L⁻¹ ≥ 0`, hence for `b > 0` every
/// `y = L⁻¹b > 0` and no cancellation hides a dropped coupling.
///
/// Bound: each computed triangular solve satisfies `(L + ΔL)ŷ = b` with
/// `|ΔL| ≤ γₙ|L|` (Higham, *Accuracy and Stability of Numerical Algorithms*,
/// 2nd ed., Thm 8.5), so its relative forward error is at most
/// `γₙ · ‖|L⁻¹||L||y|‖∞ / ‖y‖∞ ≤ γₙ · ‖L⁻¹‖∞ · ‖L‖∞ = 3γₙ`. Backward
/// substitution inherits the forward pass's error through `‖L⁻ᵀ‖∞ ‖Lᵀ‖∞ ≤ 3`
/// and adds its own `3γₙ`, giving `12γₙ` per implementation; the device and
/// host results, each within `12γₙ` of the exact solution of the shared
/// factor, differ by at most `24γₙ` with `γₙ = n·u / (1 − n·u)`, `u = 2⁻²⁴`.
/// Dropping one trailing update moves rows past the first block by
/// `Σ_c P[r,c]·y[c] ≳ 0.2·min y` — three orders above the bound.
#[cfg(feature = "decomposition")]
pub(crate) fn blocked_cholesky_solve_multi_panel_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, cholesky_decompose_blocked};
    use leto::Layout;

    let n = 300usize;
    // Deterministic LCG (Numerical Recipes constants) mapped to [0, 1).
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f64 / (1u64 << 24) as f64
    };
    let mut lower = vec![0.0f64; n * n];
    for row in 0..n {
        lower[row * n + row] = 1.0;
        for col in 0..row {
            lower[row * n + col] = -unit() / (2.0 * n as f64);
        }
    }
    let matrix_host = (0..n * n)
        .map(|index| {
            let (row, col) = (index / n, index % n);
            (0..=row.min(col))
                .map(|k| lower[row * n + k] * lower[col * n + k])
                .sum::<f64>() as f32
        })
        .collect::<Vec<_>>();
    let rhs_host = (0..n).map(|_| 1.0 + unit() as f32).collect::<Vec<_>>();

    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_cholesky = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    assert!(
        !gpu_cholesky.host_factor_materialized(),
        "the blocked factorization must not download its factor"
    );

    let solution = gpu_cholesky.solve(&device, &rhs).unwrap();
    assert!(
        !gpu_cholesky.host_factor_materialized(),
        "a device solve must not materialize the host factor"
    );
    let mut got = vec![0.0f32; n];
    device.download(&solution, &mut got).unwrap();
    let mut rhs_after = vec![0.0f32; n];
    device.download(&rhs, &mut rhs_after).unwrap();
    assert_eq!(
        rhs_after, rhs_host,
        "the caller's right-hand side is read-only"
    );

    let mut factor = vec![0.0f32; n * n];
    device.download(gpu_cholesky.lower(), &mut factor).unwrap();
    let host_factor = leto_ops::CholeskyDecomposition::from_raw_parts(
        leto::Array2::from_shape_vec([n, n], factor).unwrap(),
    );
    let rhs_view = leto::ArrayView::<f32, 1>::new(Layout::c_contiguous([n]).unwrap(), &rhs_host);
    let expected = host_factor.solve(&rhs_view).unwrap();
    let expected = leto::Storage::as_slice(expected.storage());

    let n_u = n as f32 * f32::EPSILON / 2.0;
    let gamma_n = n_u / (1.0 - n_u);
    let bound = 24.0 * gamma_n;
    let scale = expected.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
    for (row, (&device_x, &host_x)) in got.iter().zip(expected).enumerate() {
        assert!(
            (device_x - host_x).abs() <= bound * scale,
            "x[{row}]: device {device_x} vs leto {host_x}, bound {} (relative {bound})",
            bound * scale
        );
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn udu_decompose_rejects_invalid_contracts() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, udu_decompose};
    use leto::Layout;

    let rectangular_host = vec![1.0f32, 2.0, 3.0, 2.0, 4.0, 5.0];
    let rectangular = device.upload(&rectangular_host).unwrap();
    let rectangular_layout = Layout::c_contiguous([2, 3]).unwrap();
    let rectangular_result = udu_decompose(
        &device,
        StridedOperand {
            buffer: &rectangular,
            layout: &rectangular_layout,
        },
    );
    assert!(matches!(
        rectangular_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));

    let nonsymmetric_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let nonsymmetric = device.upload(&nonsymmetric_host).unwrap();
    let nonsymmetric_layout = Layout::c_contiguous([2, 2]).unwrap();
    let nonsymmetric_result = udu_decompose(
        &device,
        StridedOperand {
            buffer: &nonsymmetric,
            layout: &nonsymmetric_layout,
        },
    );
    assert!(matches!(
        nonsymmetric_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("UDU decomposition failed")
    ));

    let zero_pivot_host = vec![1.0f32, 1.0, 1.0, 0.0];
    let zero_pivot = device.upload(&zero_pivot_host).unwrap();
    let zero_pivot_layout = Layout::c_contiguous([2, 2]).unwrap();
    let zero_pivot_result = udu_decompose(
        &device,
        StridedOperand {
            buffer: &zero_pivot,
            layout: &zero_pivot_layout,
        },
    );
    assert!(matches!(
        zero_pivot_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("UDU decomposition failed")
    ));
}

#[cfg(feature = "decomposition")]
pub(crate) fn bunch_kaufman_rejects_rectangular_and_nonsymmetric() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, bunch_kaufman};
    use leto::Layout;

    let rectangular_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rectangular = device.upload(&rectangular_host).unwrap();
    let rectangular_layout = Layout::c_contiguous([2, 3]).unwrap();
    let rectangular_result = bunch_kaufman(
        &device,
        StridedOperand {
            buffer: &rectangular,
            layout: &rectangular_layout,
        },
    );
    assert!(matches!(
        rectangular_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));

    let nonsymmetric_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let nonsymmetric = device.upload(&nonsymmetric_host).unwrap();
    let nonsymmetric_layout = Layout::c_contiguous([2, 2]).unwrap();
    let nonsymmetric_result = bunch_kaufman(
        &device,
        StridedOperand {
            buffer: &nonsymmetric,
            layout: &nonsymmetric_layout,
        },
    );
    assert!(matches!(
        nonsymmetric_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Bunch-Kaufman decomposition failed")
    ));
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_cholesky_rejects_non_dense_operands() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_blocked_rejects_non_dense(
        &device,
        hephaestus_wgpu::cholesky_decompose_blocked,
        "cholesky",
    );
}
