#[cfg(feature = "decomposition")]
#[test]
fn symmetric_eigen_surfaces_match_leto_and_reconstruct() {
    let Some(device) = device("symmetric_eigen_surfaces_match_leto_and_reconstruct") else {
        return;
    };

    let matrix_values = [6.0_f32, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP symmetric eigen upload");
    let layout = Layout::c_contiguous([3, 3]).expect("symmetric eigen layout");
    let factor = symmetric_eigen_jacobi(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm symmetric eigendecomposition");
    let expected =
        leto_ops::symmetric_eigen_jacobi(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
            .expect("CPU symmetric eigendecomposition reference");

    assert_eq!(factor.n(), 3);
    let mut actual_values = [0.0_f32; 3];
    let mut actual_vectors = [0.0_f32; 9];
    device
        .download(factor.eigenvalues(), &mut actual_values)
        .expect("HIP symmetric eigenvalues download");
    device
        .download(factor.eigenvectors(), &mut actual_vectors)
        .expect("HIP symmetric eigenvectors download");
    for (actual, expected) in actual_values.iter().zip(&expected.eigenvalues) {
        assert_near(*actual, *expected, 4096.0);
    }
    for row in 0..3 {
        for col in 0..3 {
            let value = (0..3)
                .map(|k| matrix_values[row * 3 + k] * actual_vectors[k * 3 + col])
                .sum();
            assert_near(
                value,
                actual_values[col] * actual_vectors[row * 3 + col],
                8192.0,
            );
        }
    }

    let values_only = symmetric_eigenvalues_jacobi(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm symmetric eigenvalues-only");
    let mut actual_values_only = [0.0_f32; 3];
    device
        .download(&values_only, &mut actual_values_only)
        .expect("HIP symmetric eigenvalues-only download");
    assert_eq!(actual_values, actual_values_only);

    let nonsymmetric = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("HIP nonsymmetric eigen upload");
    let nonsymmetric_layout = Layout::c_contiguous([2, 2]).expect("nonsymmetric eigen layout");
    assert!(matches!(
        symmetric_eigen_jacobi(
            &device,
            StridedOperand {
                buffer: &nonsymmetric,
                layout: &nonsymmetric_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("not symmetric")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn general_eigenvalues_match_leto_complex_spectrum_and_validate_inputs() {
    let Some(device) =
        device("general_eigenvalues_match_leto_complex_spectrum_and_validate_inputs")
    else {
        return;
    };

    let matrix_values = [1.0_f32, -1.0, 1.0, 1.0];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP general eigen upload");
    let layout = Layout::c_contiguous([2, 2]).expect("general eigen layout");
    let actual = eigenvalues(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm general eigenvalues");
    let mut actual_values = [eunomia::Complex::new(0.0_f32, 0.0); 2];
    device
        .download(&actual, &mut actual_values)
        .expect("HIP general eigenvalues download");
    let expected = leto_ops::eigenvalues(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU general eigenvalues reference");
    assert_complex_spectrum_close(&actual_values, &expected, 4096.0);

    let nonsquare = device
        .upload(&[1.0_f32; 6])
        .expect("HIP nonsquare eigen upload");
    let nonsquare_layout = Layout::c_contiguous([2, 3]).expect("nonsquare eigen layout");
    assert!(matches!(
        eigenvalues(
            &device,
            StridedOperand {
                buffer: &nonsquare,
                layout: &nonsquare_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("square")
    ));

    let nonfinite = device
        .upload(&[1.0_f32, 0.0, 0.0, f32::NAN])
        .expect("HIP non-finite eigen upload");
    assert!(matches!(
        eigenvalues(
            &device,
            StridedOperand {
                buffer: &nonfinite,
                layout: &layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("non-finite")
    ));

    let empty = device
        .upload(&[] as &[f32])
        .expect("HIP empty eigen upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty eigen layout");
    let empty_symmetric = symmetric_eigen_jacobi(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("ROCm empty symmetric eigen");
    assert_eq!(empty_symmetric.n(), 0);
    let empty_general = eigenvalues(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("ROCm empty general eigen");
    assert_eq!(empty_general.len(), 0);
}

#[cfg(feature = "decomposition")]
#[test]
fn hessenberg_matches_leto_and_reconstructs_similarity() {
    let Some(device) = device("hessenberg_matches_leto_and_reconstructs_similarity") else {
        return;
    };

    let matrix_values = [
        4.0_f32, 5.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0,
    ];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP Hessenberg upload");
    let layout = Layout::c_contiguous([4, 4]).expect("Hessenberg layout");
    let factor = hessenberg(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm Hessenberg reduction");
    let expected = leto_ops::hessenberg(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU Hessenberg reference");

    assert_eq!(factor.n(), 4);
    let expected_q_matrix = expected.q();
    let expected_h_matrix = expected.h();
    let expected_q = leto::Storage::as_slice(expected_q_matrix.storage());
    let expected_h = leto::Storage::as_slice(expected_h_matrix.storage());
    let mut actual_q = [0.0_f32; 16];
    let mut actual_h = [0.0_f32; 16];
    device
        .download(factor.q_buffer(), &mut actual_q)
        .expect("HIP Hessenberg Q download");
    device
        .download(factor.h_buffer(), &mut actual_h)
        .expect("HIP Hessenberg H download");
    for (actual, expected) in actual_q.iter().zip(expected_q) {
        assert_near(*actual, *expected, 8192.0);
    }
    for (actual, expected) in actual_h.iter().zip(expected_h) {
        assert_near(*actual, *expected, 8192.0);
    }

    assert_orthogonal(&actual_q, 4);
    for row in 0..4 {
        for col in 0..4 {
            if row > col + 1 {
                assert_near(actual_h[row * 4 + col], 0.0, 8192.0);
            }
        }
    }
    let qh = matmul_square(&actual_q, &actual_h, 4);
    let reconstructed = matmul_square(&qh, &transpose_square(&actual_q, 4), 4);
    for (actual, expected) in reconstructed.iter().zip(matrix_values) {
        assert_near(*actual, expected, 16384.0);
    }

    let empty = device
        .upload(&[] as &[f32])
        .expect("HIP empty Hessenberg upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty Hessenberg layout");
    let empty_factor = hessenberg(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("ROCm empty Hessenberg");
    assert_eq!(empty_factor.n(), 0);
    assert_eq!(empty_factor.q_buffer().len(), 0);
    assert_eq!(empty_factor.h_buffer().len(), 0);
}

#[cfg(feature = "decomposition")]
#[test]
fn hessenberg_rejects_rectangular_and_nonfinite_inputs() {
    let Some(device) = device("hessenberg_rejects_rectangular_and_nonfinite_inputs") else {
        return;
    };

    let rectangular = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("HIP rectangular Hessenberg upload");
    let rectangular_layout = Layout::c_contiguous([2, 3]).expect("rectangular Hessenberg layout");
    assert!(matches!(
        hessenberg(
            &device,
            StridedOperand {
                buffer: &rectangular,
                layout: &rectangular_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square")
    ));

    let nonfinite = device
        .upload(&[1.0_f32, 0.0, 0.0, f32::NAN])
        .expect("HIP nonfinite Hessenberg upload");
    let square_layout = Layout::c_contiguous([2, 2]).expect("square Hessenberg layout");
    assert!(matches!(
        hessenberg(
            &device,
            StridedOperand {
                buffer: &nonfinite,
                layout: &square_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("non-finite")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn schur_matches_leto_and_reconstructs_real_spectrum() {
    let Some(device) = device("schur_matches_leto_and_reconstructs_real_spectrum") else {
        return;
    };

    let matrix_values = [1.0_f32, -3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 5.0];
    let matrix = device.upload(&matrix_values).expect("HIP Schur upload");
    let layout = Layout::c_contiguous([3, 3]).expect("Schur layout");
    let factor = schur(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm real Schur decomposition");
    let expected = leto_ops::schur(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU Schur reference");

    assert_eq!(factor.n(), 3);
    let expected_q_matrix = expected.q();
    let expected_t_matrix = expected.t();
    let expected_q = leto::Storage::as_slice(expected_q_matrix.storage());
    let expected_t = leto::Storage::as_slice(expected_t_matrix.storage());
    let mut actual_q = [0.0_f32; 9];
    let mut actual_t = [0.0_f32; 9];
    device
        .download(factor.q_buffer(), &mut actual_q)
        .expect("HIP Schur Q download");
    device
        .download(factor.t_buffer(), &mut actual_t)
        .expect("HIP Schur T download");
    for (actual, expected) in actual_q.iter().zip(expected_q) {
        assert_near(*actual, *expected, 8192.0);
    }
    for (actual, expected) in actual_t.iter().zip(expected_t) {
        assert_near(*actual, *expected, 8192.0);
    }

    assert_orthogonal(&actual_q, 3);
    for row in 0..3 {
        for col in 0..3 {
            if row > col + 1 {
                assert_near(actual_t[row * 3 + col], 0.0, 8192.0);
            }
        }
    }
    let subdiagonal = actual_t[3];
    assert!(subdiagonal.abs() > 8192.0 * f32::EPSILON);
    let discriminant = (actual_t[0] - actual_t[4]).powi(2) + 4.0 * actual_t[1] * subdiagonal;
    assert!(
        discriminant < 0.0,
        "expected a real Schur 2x2 complex block"
    );

    let qt = matmul_square(&actual_q, &actual_t, 3);
    let reconstructed = matmul_square(&qt, &transpose_square(&actual_q, 3), 3);
    for (actual, expected) in reconstructed.iter().zip(matrix_values) {
        assert_near(*actual, expected, 16384.0);
    }

    let t_buffer = device
        .upload(&actual_t)
        .expect("HIP Schur T spectrum upload");
    let t_values = eigenvalues(
        &device,
        StridedOperand {
            buffer: &t_buffer,
            layout: &layout,
        },
    )
    .expect("ROCm Schur T spectrum");
    let a_values = eigenvalues(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm input spectrum");
    let mut actual_t_values = [eunomia::Complex::new(0.0_f32, 0.0); 3];
    let mut actual_a_values = [eunomia::Complex::new(0.0_f32, 0.0); 3];
    device
        .download(&t_values, &mut actual_t_values)
        .expect("HIP Schur T spectrum download");
    device
        .download(&a_values, &mut actual_a_values)
        .expect("HIP input spectrum download");
    assert_complex_spectrum_close(&actual_t_values, &actual_a_values, 8192.0);

    let empty = device
        .upload(&[] as &[f32])
        .expect("HIP empty Schur upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty Schur layout");
    let empty_factor = schur(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("ROCm empty Schur");
    assert_eq!(empty_factor.n(), 0);
    assert_eq!(empty_factor.q_buffer().len(), 0);
    assert_eq!(empty_factor.t_buffer().len(), 0);
}

#[cfg(feature = "decomposition")]
#[test]
fn schur_rejects_rectangular_and_nonfinite_inputs() {
    let Some(device) = device("schur_rejects_rectangular_and_nonfinite_inputs") else {
        return;
    };

    let rectangular = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("HIP rectangular Schur upload");
    let rectangular_layout = Layout::c_contiguous([2, 3]).expect("rectangular Schur layout");
    assert!(matches!(
        schur(
            &device,
            StridedOperand {
                buffer: &rectangular,
                layout: &rectangular_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square")
    ));

    let nonfinite = device
        .upload(&[1.0_f32, 0.0, 0.0, f32::NAN])
        .expect("HIP nonfinite Schur upload");
    let square_layout = Layout::c_contiguous([2, 2]).expect("square Schur layout");
    assert!(matches!(
        schur(
            &device,
            StridedOperand {
                buffer: &nonfinite,
                layout: &square_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("non-finite")
    ));
}
