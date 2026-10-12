#[cfg(feature = "decomposition")]
use super::*;

#[cfg(feature = "decomposition")]
#[test]
fn lu_factorization_and_host_contracts_match_values() {
    let Some(device) = device("lu_factorization_and_host_contracts_match_values") else {
        return;
    };

    let matrix_values = [0.0_f32, 2.0, 1.0, 3.0];
    let matrix = device.upload(&matrix_values).expect("HIP LU input upload");
    let layout = Layout::c_contiguous([2, 2]).expect("LU matrix layout");
    let factor = lu_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP LU factorization");

    assert_eq!(factor.n(), 2);
    assert_eq!(factor.pivots(), &[1, 1]);
    assert_near(factor.det(), -2.0, 128.0);
    let mut packed = [0.0_f32; 4];
    device
        .download(factor.factors(), &mut packed)
        .expect("HIP LU factor download");
    assert_near(packed[0], 1.0, 128.0);
    assert_near(packed[1], 3.0, 128.0);
    assert_near(packed[2], 0.0, 128.0);
    assert_near(packed[3], 2.0, 128.0);

    let rhs = device.upload(&[4.0_f32, 7.0]).expect("HIP LU RHS upload");
    let solution = factor.solve(&device, &rhs).expect("HIP LU solve");
    let mut solution_values = [0.0_f32; 2];
    device
        .download(&solution, &mut solution_values)
        .expect("HIP LU solution download");
    assert_near(solution_values[0], 1.0, 128.0);
    assert_near(solution_values[1], 2.0, 128.0);

    let inverse = factor.inv(&device).expect("HIP LU inverse");
    let mut inverse_values = [0.0_f32; 4];
    device
        .download(&inverse, &mut inverse_values)
        .expect("HIP LU inverse download");
    for (actual, expected) in inverse_values.iter().zip([-1.5_f32, 1.0, 0.5, 0.0]) {
        assert_near(*actual, expected, 128.0);
    }

    let blocked = lu_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP blocked LU factorization");
    assert_eq!(blocked.pivots(), factor.pivots());
    let mut blocked_packed = [0.0_f32; 4];
    device
        .download(blocked.factors(), &mut blocked_packed)
        .expect("HIP blocked LU factor download");
    for (actual, expected) in blocked_packed.iter().zip(packed) {
        assert_near(*actual, expected, 128.0);
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn lu_strided_and_invalid_contracts_are_enforced() {
    let Some(device) = device("lu_strided_and_invalid_contracts_are_enforced") else {
        return;
    };

    let strided_values = [0.0_f32, 2.0, 99.0, 1.0, 3.0, 88.0];
    let strided_buffer = device
        .upload(&strided_values)
        .expect("HIP strided LU upload");
    let strided_layout = Layout::try_new([2, 2], [3, 1], 0).expect("valid test layout");
    let factor = lu_decompose(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
    )
    .expect("HIP strided LU factorization");
    assert_eq!(factor.pivots(), &[1, 1]);

    let blocked = lu_decompose_blocked(
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

    let singular_buffer = device
        .upload(&[1.0_f32, 2.0, 2.0, 4.0])
        .expect("HIP singular LU upload");
    let dense_layout = Layout::c_contiguous([2, 2]).expect("dense LU layout");
    assert!(matches!(
        lu_decompose(
            &device,
            StridedOperand {
                buffer: &singular_buffer,
                layout: &dense_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("singular")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn full_piv_lu_matches_device_permutations_and_host_contracts() {
    let Some(device) = device("full_piv_lu_matches_device_permutations_and_host_contracts") else {
        return;
    };

    let matrix_values = [0.0_f32, 2.0, 1.0, 3.0];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP complete-pivot LU upload");
    let layout = Layout::c_contiguous([2, 2]).expect("complete-pivot LU layout");
    let factor = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP complete-pivot LU factorization");
    let expected = leto_ops::full_piv_lu(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU complete-pivot LU reference");

    assert_eq!(factor.n(), 2);
    assert_eq!(factor.rank(), expected.rank());
    assert_eq!(factor.row_permutation(), expected.row_permutation());
    assert_eq!(factor.col_permutation(), expected.col_permutation());
    assert_near(factor.det(), expected.det(), 256.0);

    let mut packed = [0.0_f32; 4];
    device
        .download(factor.lu_buffer(), &mut packed)
        .expect("HIP complete-pivot LU factor download");
    for (actual, expected) in packed.iter().zip(expected.lu_factors()) {
        assert_near(*actual, *expected, 512.0);
    }

    let rhs = device
        .upload(&[4.0_f32, 7.0])
        .expect("HIP complete-pivot LU RHS upload");
    let solution = factor
        .solve(&device, &rhs)
        .expect("HIP complete-pivot LU solve");
    let mut solution_values = [0.0_f32; 2];
    device
        .download(&solution, &mut solution_values)
        .expect("HIP complete-pivot LU solution download");
    assert_near(solution_values[0], 1.0, 512.0);
    assert_near(solution_values[1], 2.0, 512.0);

    let inverse = factor.inv(&device).expect("HIP complete-pivot LU inverse");
    let mut inverse_values = [0.0_f32; 4];
    device
        .download(&inverse, &mut inverse_values)
        .expect("HIP complete-pivot LU inverse download");
    for (actual, expected) in inverse_values.iter().zip([-1.5_f32, 1.0, 0.5, 0.0]) {
        assert_near(*actual, expected, 512.0);
    }

    let blocked = full_piv_lu_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("HIP dense complete-pivot LU factorization");
    assert_eq!(blocked.row_permutation(), factor.row_permutation());
    assert_eq!(blocked.col_permutation(), factor.col_permutation());
}

#[cfg(feature = "decomposition")]
#[test]
fn full_piv_lu_preserves_strided_and_rank_contracts() {
    let Some(device) = device("full_piv_lu_preserves_strided_and_rank_contracts") else {
        return;
    };

    let strided_values = [0.0_f32, 2.0, 99.0, 1.0, 3.0, 88.0];
    let strided_buffer = device
        .upload(&strided_values)
        .expect("HIP strided complete-pivot LU upload");
    let strided_layout = Layout::try_new([2, 2], [3, 1], 0).expect("valid test layout");
    let factor = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
    )
    .expect("HIP strided complete-pivot LU factorization");
    assert_eq!(factor.rank(), 2);

    let dense = full_piv_lu_blocked(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
    );
    assert!(matches!(
        dense,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("dense") || message.contains("contiguous")
    ));

    let singular = device
        .upload(&[1.0_f32, 2.0, 2.0, 4.0])
        .expect("HIP singular complete-pivot LU upload");
    let layout = Layout::c_contiguous([2, 2]).expect("singular complete-pivot LU layout");
    let factor = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &singular,
            layout: &layout,
        },
    )
    .expect("rank-deficient complete-pivot LU remains factorable");
    assert_eq!(factor.rank(), 1);
    assert_eq!(factor.det(), 0.0);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_reject_non_dense_operands() {
    let Some(device) = device("blocked_pivoted_decompositions_reject_non_dense_operands") else {
        return;
    };
    let dense = device
        .upload(&[1.0_f32; 16])
        .expect("HIP dense blocked pivoted input upload");
    let small = device
        .upload(&[1.0_f32; 4])
        .expect("HIP broadcast blocked pivoted input upload");
    let transposed = Layout::try_new([4, 4], [1, 4], 0).expect("valid test layout");
    let broadcast = Layout::try_new([4, 4], [0, 1], 0).expect("valid test layout");

    let lu_result = full_piv_lu_blocked(
        &device,
        StridedOperand {
            buffer: &dense,
            layout: &transposed,
        },
    );
    assert!(matches!(
        lu_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("dense") || message.contains("contiguous")
    ));

    let qr_result = col_piv_qr_blocked(
        &device,
        StridedOperand {
            buffer: &small,
            layout: &broadcast,
        },
    );
    assert!(matches!(
        qr_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("dense") || message.contains("contiguous")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn fluent_decomposition_traits_match_solve_contract() {
    let Some(device) = device("fluent_decomposition_traits_match_solve_contract") else {
        return;
    };

    let values = [4.0_f32, 1.0, 2.0, 3.0];
    let matrix = device.upload(&values).expect("HIP fluent LU upload");
    let layout = Layout::c_contiguous([2, 2]).expect("fluent LU layout");
    let operand = StridedOperand {
        buffer: &matrix,
        layout: &layout,
    };
    let factor = operand.lu(&device).expect("ROCm fluent LU");
    let rhs = device
        .upload(&[1.0_f32, 2.0])
        .expect("HIP fluent rhs upload");
    let solution = factor.solve(&device, &rhs).expect("ROCm fluent LU solve");
    let mut actual = [0.0_f32; 2];
    device
        .download(&solution, &mut actual)
        .expect("HIP fluent solve download");
    assert_near(actual[0], 0.1, 4096.0);
    assert_near(actual[1], 0.6, 4096.0);
}
