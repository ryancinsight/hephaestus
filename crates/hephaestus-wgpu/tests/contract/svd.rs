use super::*;

pub(crate) fn linalg_matrix_rank_rejects_empty_operands() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    let empty = device.alloc_zeroed::<f32>(1).unwrap();
    let empty_layout = Layout::c_contiguous([0, 3]).unwrap();
    let result = matrix_rank(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    );
    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message }) if message.contains("empty matrix")
    ));
}

/// Pins the GPU `matrix_rank` relative-threshold contract at the boundary that
/// the residual register flagged as the ill-conditioned divergence risk: a pivot
/// counts toward the rank iff its magnitude exceeds
/// `relative_tolerance * max(abs(matrix))`. For `diag(1, 1, δ)` the max element
/// is `1` and the singular values equal the diagonal magnitudes, so the
/// threshold alone decides the rank and the GPU row-reduction result must agree
/// with Leto's SVD-spectrum result for both tolerances.
pub(crate) fn matrix_rank_relative_tolerance_is_the_discriminator() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    // diag(1, 1, 1e-4): max abs = 1, smallest singular value = 1e-4.
    let host = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-4];
    let buffer = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], host).unwrap();

    let tol_keep = 1.0e-6f32; // 1e-4 > 1e-6 * 1 -> small pivot retained
    let tol_drop = 1.0e-2f32; // 1e-4 < 1e-2 * 1 -> small pivot dropped

    let operand = |buffer| StridedOperand {
        buffer,
        layout: &layout,
    };
    let rank_keep = matrix_rank_with_tolerance(&device, operand(&buffer), tol_keep).unwrap();
    let rank_drop = matrix_rank_with_tolerance(&device, operand(&buffer), tol_drop).unwrap();

    assert_eq!(
        rank_keep, 3,
        "1e-4 > 1e-6 * max_abs must count the small pivot"
    );
    assert_eq!(
        rank_drop, 2,
        "1e-4 < 1e-2 * max_abs must drop the small pivot"
    );

    // The diagonal case has singular values == pivot magnitudes, so Leto's
    // SVD-spectrum criterion agrees with the GPU row-reduction threshold.
    let leto_keep = leto_ops::matrix_rank_with_tolerance(&leto_matrix.view(), tol_keep).unwrap();
    let leto_drop = leto_ops::matrix_rank_with_tolerance(&leto_matrix.view(), tol_drop).unwrap();
    assert_eq!(rank_keep, leto_keep);
    assert_eq!(rank_drop, leto_drop);
}

/// Pins the GPU `det` contract on an ill-conditioned (tiny-pivot) matrix: the
/// kernel returns the row-reduction pivot product with no determinant-tolerance
/// zeroing (`det` passes `tolerance == 0`, so only exactly-zero pivots drop), so
/// a near-singular matrix yields its small, nonzero determinant. The input is
/// upper-triangular, so elimination performs no row operations and the result is
/// the analytical pivot product `2 * 3 * δ`.
pub(crate) fn det_of_near_singular_triangular_is_exact_pivot_product() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, det};
    use leto::Layout;

    let delta = 1.0e-5f32;
    let host = vec![2.0f32, 1.0, 5.0, 0.0, 3.0, 7.0, 0.0, 0.0, delta];
    let buffer = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], host).unwrap();

    let det_buf = det(
        &device,
        StridedOperand {
            buffer: &buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got = [0.0f32; 1];
    device.download(&det_buf, &mut got).unwrap();

    // Triangular elimination performs no row operations, so the only error is
    // the f32 rounding of `delta` and the pivot product; a few ulps bound it.
    let analytical = 6.0f32 * delta;
    let tol = analytical.abs() * 8.0 * f32::EPSILON;
    assert!(
        (got[0] - analytical).abs() <= tol,
        "GPU det {} must match analytical pivot product {analytical} within {tol}",
        got[0]
    );
    // The determinant must NOT be zeroed despite being tiny (no det tolerance).
    assert!(
        got[0] > 0.0,
        "near-singular det must stay strictly positive, got {}",
        got[0]
    );
    // Differential: Leto's determinant agrees on this triangular input.
    let leto_det = leto_ops::det(&leto_matrix.view()).unwrap();
    assert!((got[0] - leto_det).abs() <= tol);
}

#[cfg(feature = "decomposition")]
pub(crate) fn singular_values_match_closed_form_diagonal() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, singular_values};
    use leto::Layout;

    let matrix_host = vec![3.0f32, 0.0, 0.0, 0.0, 2.0, 0.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let values = singular_values(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 2];
    device.download(&values, &mut got).unwrap();
    assert_eq!(got.len(), 2);
    assert_close(got[0], 3.0, 1.0e-5);
    assert_close(got[1], 2.0, 1.0e-5);
}

#[cfg(feature = "decomposition")]
pub(crate) fn svd_decompose_reconstructs_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, svd_decompose};
    use leto::Layout;

    let rows = 4usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_svd.shape(), (rows, cols));
    let leto_matrix = leto::Array::from_shape_vec([rows, cols], matrix_host.clone()).unwrap();
    let leto_svd = leto_ops::svd_decompose(&leto_matrix.view()).unwrap();

    let rank = leto_svd.singular_values.len();
    let mut got_singular = vec![0.0f32; rank];
    let mut got_u = vec![0.0f32; rows * rank];
    let mut got_v = vec![0.0f32; cols * rank];
    device
        .download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();
    device.download(gpu_svd.u(), &mut got_u).unwrap();
    device.download(gpu_svd.v(), &mut got_v).unwrap();

    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-5);
    }

    let reconstructed = reconstruct_svd(&got_u, &got_singular, &got_v, rows, cols);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn svd_rank_revealing_accepts_rank_deficient_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, svd_rank_revealing};
    use leto::Layout;

    let rows = 3usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_rank_revealing(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let leto_matrix = leto::Array::from_shape_vec([rows, cols], matrix_host).unwrap();
    let leto_svd = leto_ops::svd_decompose(&leto_matrix.view()).unwrap();
    let rank = leto_svd.singular_values.len();
    let mut got_singular = vec![0.0f32; rank];
    device
        .download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();

    assert_eq!(rank, 2);
    assert!(got_singular[0] >= got_singular[1]);
    assert_close(got_singular[1], 0.0, 1.0e-5);
    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn bidiagonalize_reconstructs_and_preserves_singular_values() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, bidiagonalize, singular_values};
    use leto::Layout;

    let rows = 4usize;
    let cols = 3usize;
    let matrix_host = vec![
        4.0f32, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
    ];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_bd = bidiagonalize(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_bd.shape(), (rows, cols));
    let mut u = vec![0.0f32; rows * rows];
    let mut b = vec![0.0f32; rows * cols];
    let mut v = vec![0.0f32; cols * cols];
    device.download(gpu_bd.u_buffer(), &mut u).unwrap();
    device.download(gpu_bd.b_buffer(), &mut b).unwrap();
    device.download(gpu_bd.v_buffer(), &mut v).unwrap();

    assert_orthogonal_host(&u, rows, 1.0e-4);
    assert_orthogonal_host(&v, cols, 1.0e-4);
    for row in 0..rows {
        for col in 0..cols {
            if col < row || col > row + 1 {
                assert_close(b[row * cols + col], 0.0, 1.0e-4);
            }
        }
    }

    let ub = matmul_host(&u, rows, rows, &b, cols);
    let vt = transpose_host(&v, cols, cols);
    let reconstructed = matmul_host(&ub, rows, cols, &vt, cols);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-3);
    }

    let b_buffer = device.upload(&b).unwrap();
    let b_layout = Layout::c_contiguous([rows, cols]).unwrap();
    let sv_b = singular_values(
        &device,
        StridedOperand {
            buffer: &b_buffer,
            layout: &b_layout,
        },
    )
    .unwrap();
    let sv_a = singular_values(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_b = vec![0.0f32; cols];
    let mut got_a = vec![0.0f32; cols];
    device.download(&sv_b, &mut got_b).unwrap();
    device.download(&sv_a, &mut got_a).unwrap();
    for (actual, expected) in got_b.iter().zip(got_a.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn full_piv_lu_reveals_rank_deficiency_and_rejects_inverse() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, full_piv_lu};
    use leto::Layout;

    let n = 3usize;
    let matrix_host = vec![1.0f32, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_lu = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.rank(), 2);
    assert_close(gpu_lu.det(), 0.0, 1.0e-5);
    assert!(matches!(
        gpu_lu.inv(&device),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("FullPivLU inverse failed")
    ));
}

#[cfg(feature = "decomposition")]
pub(crate) fn cholesky_rejects_singular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, cholesky_decompose};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert_rejects(
        &result,
        "kernel dispatch failed: Cholesky decomposition failed: Storage error: Cholesky pivot 0 is non-positive: matrix is not positive-definite",
    );
}

#[cfg(feature = "decomposition")]
pub(crate) fn lu_rejects_singular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, lu_decompose};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = lu_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert_rejects(
        &result,
        "kernel dispatch failed: LU decomposition failed: Storage error: LU pivot column 0 is exactly zero: matrix is singular",
    );
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_cholesky_rejects_singular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, cholesky_decompose_blocked};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert_rejects(
        &result,
        "kernel dispatch failed: Cholesky panel factorisation failed: pivot 0 is not positive (0); matrix is not positive-definite",
    );
}
