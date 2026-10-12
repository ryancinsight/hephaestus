#[cfg(feature = "decomposition")]
use super::*;

#[cfg(feature = "decomposition")]
#[test]
fn qr_factorization_and_least_squares_match_values() {
    let Some(device) = device("qr_factorization_and_least_squares_match_values") else {
        return;
    };

    let matrix_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&matrix_values).expect("HIP QR input upload");
    let layout = Layout::c_contiguous([3, 2]).expect("QR matrix layout");
    let factor = qr_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP QR factorization");
    assert_eq!(factor.shape(), (3, 2));

    let expected_layout = Layout::c_contiguous([3, 2]).expect("QR reference layout");
    let expected = leto_ops::qr_decompose(&leto::ArrayView::<f32, 2>::new(
        expected_layout,
        &matrix_values,
    ))
    .expect("CPU QR reference");
    let expected_r_matrix = expected.r();
    let expected_r = leto::Storage::as_slice(expected_r_matrix.storage());
    let mut actual_r = [0.0_f32; 6];
    device
        .download(factor.r_buffer(), &mut actual_r)
        .expect("HIP QR factor download");
    for row in 0..3 {
        for col in row..2 {
            assert_near(actual_r[row * 2 + col], expected_r[row * 2 + col], 256.0);
        }
    }

    let rhs = device
        .upload(&[5.0_f32, 11.0, 17.0])
        .expect("HIP QR RHS upload");
    let solution = factor
        .solve_least_squares(&device, &rhs)
        .expect("HIP QR least-squares solve");
    let mut solution_values = [0.0_f32; 2];
    device
        .download(&solution, &mut solution_values)
        .expect("HIP QR solution download");
    assert_near(solution_values[0], 1.0, 512.0);
    assert_near(solution_values[1], 2.0, 512.0);

    let blocked = qr_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP blocked QR factorization");
    assert_eq!(blocked.shape(), factor.shape());
}

#[cfg(feature = "decomposition")]
#[test]
fn col_piv_qr_matches_device_permutation_and_least_squares_contracts() {
    let Some(device) = device("col_piv_qr_matches_device_permutation_and_least_squares_contracts")
    else {
        return;
    };

    let matrix_values = [1.0_f32, 0.0, 0.0, 2.0, 0.0, 0.0];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP column-pivot QR upload");
    let layout = Layout::c_contiguous([3, 2]).expect("column-pivot QR layout");
    let factor = col_piv_qr(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP column-pivot QR factorization");
    let expected = leto_ops::col_piv_qr(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU column-pivot QR reference");
    assert_eq!(factor.rank(), expected.rank());
    assert_eq!(factor.permutation(), expected.permutation());

    let expected_r_matrix = expected.r();
    let expected_r = leto::Storage::as_slice(expected_r_matrix.storage());
    let mut actual_r = [0.0_f32; 6];
    device
        .download(factor.r(), &mut actual_r)
        .expect("HIP column-pivot QR R download");
    for row in 0..3 {
        for col in row..2 {
            assert_near(actual_r[row * 2 + col], expected_r[row * 2 + col], 1024.0);
        }
    }

    let rhs = device
        .upload(&[3.0_f32, 8.0, 0.0])
        .expect("HIP column-pivot QR RHS upload");
    let solution = factor
        .solve_least_squares(&device, &rhs)
        .expect("HIP column-pivot QR least-squares solve");
    let mut solution_values = [0.0_f32; 2];
    device
        .download(&solution, &mut solution_values)
        .expect("HIP column-pivot QR solution download");
    assert_near(solution_values[0], 3.0, 1024.0);
    assert_near(solution_values[1], 4.0, 1024.0);

    let blocked = col_piv_qr_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP dense column-pivot QR factorization");
    assert_eq!(blocked.permutation(), factor.permutation());
}

#[cfg(feature = "decomposition")]
#[test]
fn col_piv_qr_reports_rank_and_rejects_nonfinite_values() {
    let Some(device) = device("col_piv_qr_reports_rank_and_rejects_nonfinite_values") else {
        return;
    };

    let rank_deficient = device
        .upload(&[1.0_f32, 2.0, 2.0, 4.0, 3.0, 6.0])
        .expect("HIP rank-deficient column-pivot QR upload");
    let layout = Layout::c_contiguous([3, 2]).expect("rank-deficient column-pivot QR layout");
    let factor = col_piv_qr(
        &device,
        StridedOperand {
            buffer: &rank_deficient,
            layout: &layout,
        },
    )
    .expect("rank-deficient column-pivot QR factorization");
    assert_eq!(factor.rank(), 1);
    let rhs = device
        .upload(&[1.0_f32, 2.0, 3.0])
        .expect("HIP rank-deficient column-pivot QR RHS upload");
    assert!(matches!(
        factor.solve_least_squares(&device, &rhs),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("full column rank")
    ));

    let nonfinite = device
        .upload(&[1.0_f32, f32::NAN, 3.0, 4.0, 5.0, 6.0])
        .expect("HIP nonfinite column-pivot QR upload");
    assert!(matches!(
        col_piv_qr(
            &device,
            StridedOperand {
                buffer: &nonfinite,
                layout: &layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("non-finite")
    ));
}
