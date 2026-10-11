#[cfg(feature = "decomposition")]
#[test]
fn cholesky_factorization_and_common_host_contracts_match_values() {
    let Some(device) = device("cholesky_factorization_and_common_host_contracts_match_values")
    else {
        return;
    };

    let matrix_values = [4.0_f32, 1.0, 1.0, 3.0];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP Cholesky input upload");
    let layout = Layout::c_contiguous([2, 2]).expect("Cholesky matrix layout");
    let factor = cholesky_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP Cholesky factorization");

    assert_eq!(factor.n(), 2);
    let mut lower = [0.0_f32; 4];
    device
        .download(factor.lower(), &mut lower)
        .expect("HIP Cholesky factor download");
    assert_near(lower[0], 2.0, 64.0);
    assert_near(lower[1], 0.0, 64.0);
    assert_near(lower[2], 0.5, 64.0);
    assert_near(lower[3], 2.75_f32.sqrt(), 64.0);
    assert_near(factor.det(), 11.0, 128.0);

    let rhs = device
        .upload(&[1.0_f32, 2.0])
        .expect("HIP Cholesky RHS upload");
    let solution = factor.solve(&device, &rhs).expect("HIP Cholesky solve");
    let mut solution_values = [0.0_f32; 2];
    device
        .download(&solution, &mut solution_values)
        .expect("HIP Cholesky solution download");
    assert_near(solution_values[0], 1.0 / 11.0, 128.0);
    assert_near(solution_values[1], 7.0 / 11.0, 128.0);

    let inverse = factor.inv(&device).expect("HIP Cholesky inverse");
    let mut inverse_values = [0.0_f32; 4];
    device
        .download(&inverse, &mut inverse_values)
        .expect("HIP Cholesky inverse download");
    for (actual, expected) in
        inverse_values
            .iter()
            .zip([3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0])
    {
        assert_near(*actual, expected, 128.0);
    }

    let blocked = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP blocked Cholesky factorization");
    let mut blocked_lower = [0.0_f32; 4];
    device
        .download(blocked.lower(), &mut blocked_lower)
        .expect("HIP blocked Cholesky factor download");
    for (actual, expected) in blocked_lower.iter().zip(lower) {
        assert_near(*actual, expected, 64.0);
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_strided_input_and_blocked_density_contracts_are_enforced() {
    let Some(device) = device("cholesky_strided_input_and_blocked_density_contracts_are_enforced")
    else {
        return;
    };

    let strided_values = [4.0_f32, 1.0, 99.0, 1.0, 3.0, 88.0];
    let strided_buffer = device
        .upload(&strided_values)
        .expect("HIP strided Cholesky input upload");
    let strided_layout = Layout::try_new([2, 2], [3, 1], 0).expect("valid test layout");
    let factor = cholesky_decompose(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
    )
    .expect("HIP strided Cholesky factorization");
    let mut lower = [0.0_f32; 4];
    device
        .download(factor.lower(), &mut lower)
        .expect("HIP strided Cholesky factor download");
    assert_near(lower[0], 2.0, 64.0);
    assert_near(lower[2], 0.5, 64.0);

    let blocked = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
    );
    assert!(matches!(
        blocked,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("dense") || message.contains("contiguous")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_rejects_empty_invalid_and_non_positive_definite_inputs() {
    let Some(device) = device("cholesky_rejects_empty_invalid_and_non_positive_definite_inputs")
    else {
        return;
    };

    let empty_buffer = device.upload(&[] as &[f32]).expect("empty Cholesky upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty Cholesky layout");
    let empty = cholesky_decompose(
        &device,
        StridedOperand {
            buffer: &empty_buffer,
            layout: &empty_layout,
        },
    )
    .expect("empty Cholesky factorization");
    assert_eq!(empty.n(), 0);
    assert_eq!(empty.lower().len(), 0);

    let nonsquare_buffer = device
        .upload(&[1.0_f32; 6])
        .expect("nonsquare Cholesky upload");
    let nonsquare_layout = Layout::c_contiguous([2, 3]).expect("nonsquare Cholesky layout");
    assert!(matches!(
        cholesky_decompose(
            &device,
            StridedOperand {
                buffer: &nonsquare_buffer,
                layout: &nonsquare_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square")
    ));

    let nonfinite_buffer = device
        .upload(&[4.0_f32, f32::NAN, 1.0, 3.0])
        .expect("nonfinite Cholesky upload");
    let dense_layout = Layout::c_contiguous([2, 2]).expect("dense Cholesky layout");
    assert!(matches!(
        cholesky_decompose(
            &device,
            StridedOperand {
                buffer: &nonfinite_buffer,
                layout: &dense_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("non-finite")
    ));

    let indefinite_buffer = device
        .upload(&[1.0_f32, 2.0, 2.0, 1.0])
        .expect("indefinite Cholesky upload");
    assert!(matches!(
        cholesky_decompose_blocked(
            &device,
            StridedOperand {
                buffer: &indefinite_buffer,
                layout: &dense_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("not positive-definite")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn udu_matches_leto_factors_and_operations() {
    let Some(device) = device("udu_matches_leto_factors_and_operations") else {
        return;
    };

    let matrix_values = [4.0_f32, 2.0, -2.0, 2.0, -3.0, 1.0, -2.0, 1.0, 2.0];
    let matrix = device.upload(&matrix_values).expect("HIP UDU upload");
    let layout = Layout::c_contiguous([3, 3]).expect("UDU layout");
    let factor = udu_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm UDU");
    let expected = leto_ops::udu_decompose(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU UDU reference");

    assert_eq!(factor.n(), 3);
    let mut actual_u = vec![0.0_f32; 9];
    let mut actual_d = vec![0.0_f32; 3];
    device
        .download(factor.u_buffer(), &mut actual_u)
        .expect("HIP UDU U download");
    device
        .download(factor.d_buffer(), &mut actual_d)
        .expect("HIP UDU D download");
    let expected_u_matrix = expected.u();
    let expected_u = leto::Storage::as_slice(expected_u_matrix.storage());
    let expected_diagonal = expected.diagonal();
    for (actual, expected) in actual_u.iter().zip(expected_u) {
        assert_near(*actual, *expected, 4096.0);
    }
    for (actual, expected) in actual_d.iter().zip(expected_diagonal) {
        assert_near(*actual, *expected, 4096.0);
    }

    let mut reconstructed = vec![0.0_f32; matrix_values.len()];
    for row in 0..3 {
        for col in 0..3 {
            reconstructed[row * 3 + col] = (0..3)
                .map(|k| actual_u[row * 3 + k] * actual_d[k] * actual_u[col * 3 + k])
                .sum();
        }
    }
    for (actual, expected) in reconstructed.iter().zip(matrix_values) {
        assert_near(*actual, expected, 8192.0);
    }
    assert_near(factor.det(), expected.det(), 8192.0);

    let rhs = device
        .upload(&[3.0_f32, -1.0, 2.0])
        .expect("HIP UDU RHS upload");
    let solution = factor.solve(&device, &rhs).expect("ROCm UDU solve");
    let mut solution_values = [0.0_f32; 3];
    device
        .download(&solution, &mut solution_values)
        .expect("HIP UDU solution download");
    for row in 0..3 {
        let value = (0..3)
            .map(|col| matrix_values[row * 3 + col] * solution_values[col])
            .sum();
        assert_near(value, [3.0, -1.0, 2.0][row], 8192.0);
    }

    let inverse = factor.inv(&device).expect("ROCm UDU inverse");
    let mut inverse_values = [0.0_f32; 9];
    device
        .download(&inverse, &mut inverse_values)
        .expect("HIP UDU inverse download");
    for row in 0..3 {
        for col in 0..3 {
            let value = (0..3)
                .map(|k| matrix_values[row * 3 + k] * inverse_values[k * 3 + col])
                .sum();
            assert_near(value, if row == col { 1.0 } else { 0.0 }, 16384.0);
        }
    }

    let nonsquare = device
        .upload(&[1.0_f32; 6])
        .expect("HIP UDU nonsquare upload");
    let nonsquare_layout = Layout::c_contiguous([2, 3]).expect("UDU nonsquare layout");
    assert!(matches!(
        udu_decompose(
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
        .expect("HIP non-finite UDU upload");
    let nonfinite_layout = Layout::c_contiguous([2, 2]).expect("non-finite UDU layout");
    assert!(matches!(
        udu_decompose(
            &device,
            StridedOperand {
                buffer: &nonfinite,
                layout: &nonfinite_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("non-finite")
    ));

    let empty = device.upload(&[] as &[f32]).expect("HIP empty UDU upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty UDU layout");
    let empty_factor = udu_decompose(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("ROCm empty UDU");
    assert_eq!(empty_factor.n(), 0);
    assert_eq!(empty_factor.det(), 1.0);
}

#[cfg(feature = "decomposition")]
#[test]
fn bunch_kaufman_matches_leto_factors_and_permutation() {
    let Some(device) = device("bunch_kaufman_matches_leto_factors_and_permutation") else {
        return;
    };

    let matrix_values = [0.0_f32, 1.0, 1.0, 0.0];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP Bunch-Kaufman upload");
    let layout = Layout::c_contiguous([2, 2]).expect("Bunch-Kaufman layout");
    let factor = bunch_kaufman(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm Bunch-Kaufman");
    let expected = leto_ops::bunch_kaufman(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU Bunch-Kaufman reference");

    assert_eq!(factor.n(), 2);
    assert_eq!(factor.permutation(), expected.permutation());
    assert_eq!(factor.permutation(), &[1, 0]);
    let mut actual_l = vec![0.0_f32; 4];
    let mut actual_d = vec![0.0_f32; 4];
    device
        .download(factor.l_buffer(), &mut actual_l)
        .expect("HIP Bunch-Kaufman L download");
    device
        .download(factor.d_buffer(), &mut actual_d)
        .expect("HIP Bunch-Kaufman D download");
    let expected_l_matrix = expected.l();
    let expected_l = leto::Storage::as_slice(expected_l_matrix.storage());
    for (actual, expected) in actual_l.iter().zip(expected_l) {
        assert_near(*actual, *expected, 4096.0);
    }
    let expected_d_matrix = expected.d();
    let expected_d = leto::Storage::as_slice(expected_d_matrix.storage());
    for (actual, expected) in actual_d.iter().zip(expected_d) {
        assert_near(*actual, *expected, 4096.0);
    }

    let mut reconstructed = [0.0_f32; 4];
    for row in 0..2 {
        for col in 0..2 {
            let mut value = 0.0_f32;
            for p in 0..2 {
                for q in 0..2 {
                    value += actual_l[row * 2 + p] * actual_d[p * 2 + q] * actual_l[col * 2 + q];
                }
            }
            reconstructed[row * 2 + col] = value;
        }
    }
    for row in 0..2 {
        for col in 0..2 {
            let expected_value =
                matrix_values[factor.permutation()[row] * 2 + factor.permutation()[col]];
            assert_near(reconstructed[row * 2 + col], expected_value, 8192.0);
        }
    }

    let nonsymmetric = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("HIP nonsymmetric Bunch-Kaufman upload");
    assert!(matches!(
        bunch_kaufman(
            &device,
            StridedOperand {
                buffer: &nonsymmetric,
                layout: &layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("symmetric")
    ));
    let nonfinite = device
        .upload(&[1.0_f32, 0.0, 0.0, f32::NAN])
        .expect("HIP non-finite Bunch-Kaufman upload");
    assert!(matches!(
        bunch_kaufman(
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
        .expect("HIP empty Bunch-Kaufman upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty Bunch-Kaufman layout");
    let empty_factor = bunch_kaufman(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("ROCm empty Bunch-Kaufman");
    assert_eq!(empty_factor.n(), 0);
    assert!(empty_factor.permutation().is_empty());
}
