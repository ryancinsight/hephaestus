use super::*;

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_rejects_singular_matrix() {
    let Some(dev) = device("cholesky_rejects_singular_matrix") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose(
        &dev,
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
#[test]
fn lu_rejects_singular_matrix() {
    let Some(dev) = device("lu_rejects_singular_matrix") else {
        return;
    };
    use hephaestus_cuda::lu_decompose;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = lu_decompose(
        &dev,
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
#[test]
fn blocked_cholesky_rejects_singular_matrix() {
    let Some(dev) = device("blocked_cholesky_rejects_singular_matrix") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, cholesky_decompose_blocked};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose_blocked(
        &dev,
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

#[cfg(feature = "decomposition")]
#[test]
fn singular_values_match_closed_form_diagonal() {
    let Some(dev) = device("singular_values_match_closed_form_diagonal") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, singular_values};
    use leto::Layout;

    let matrix_host = vec![3.0f32, 0.0, 0.0, 0.0, 2.0, 0.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let values = singular_values(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 2];
    dev.download(&values, &mut got).unwrap();
    assert_eq!(got.len(), 2);
    assert_close(got[0], 3.0, 1.0e-5);
    assert_close(got[1], 2.0, 1.0e-5);
}

#[cfg(feature = "decomposition")]
#[test]
fn svd_decompose_reconstructs_leto_reference() {
    let Some(dev) = device("svd_decompose_reconstructs_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, svd_decompose};
    use leto::Layout;

    let rows = 4usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_decompose(
        &dev,
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
    dev.download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();
    dev.download(gpu_svd.u(), &mut got_u).unwrap();
    dev.download(gpu_svd.v(), &mut got_v).unwrap();

    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-5);
    }

    let reconstructed = reconstruct_svd(&got_u, &got_singular, &got_v, rows, cols);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn svd_rank_revealing_accepts_rank_deficient_matrix() {
    let Some(dev) = device("svd_rank_revealing_accepts_rank_deficient_matrix") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, svd_rank_revealing};
    use leto::Layout;

    let rows = 3usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_rank_revealing(
        &dev,
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
    dev.download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();

    assert_eq!(rank, 2);
    assert!(got_singular[0] >= got_singular[1]);
    assert_close(got_singular[1], 0.0, 1.0e-5);
    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn bidiagonalize_reconstructs_and_preserves_singular_values() {
    let Some(dev) = device("bidiagonalize_reconstructs_and_preserves_singular_values") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, bidiagonalize, singular_values};
    use leto::Layout;

    let rows = 4usize;
    let cols = 3usize;
    let matrix_host = vec![
        4.0f32, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
    ];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_bd = bidiagonalize(
        &dev,
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
    dev.download(gpu_bd.u_buffer(), &mut u).unwrap();
    dev.download(gpu_bd.b_buffer(), &mut b).unwrap();
    dev.download(gpu_bd.v_buffer(), &mut v).unwrap();

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

    let b_buffer = dev.upload(&b).unwrap();
    let b_layout = Layout::c_contiguous([rows, cols]).unwrap();
    let sv_b = singular_values(
        &dev,
        StridedOperand {
            buffer: &b_buffer,
            layout: &b_layout,
        },
    )
    .unwrap();
    let sv_a = singular_values(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_b = vec![0.0f32; cols];
    let mut got_a = vec![0.0f32; cols];
    dev.download(&sv_b, &mut got_b).unwrap();
    dev.download(&sv_a, &mut got_a).unwrap();
    for (actual, expected) in got_b.iter().zip(got_a.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}
