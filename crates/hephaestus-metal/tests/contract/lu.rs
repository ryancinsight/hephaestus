use super::*;

#[cfg(feature = "decomposition")]
#[test]
fn fluent_matrix_decomposition_uses_metal_selected_device() {
    let Some(device) = device("fluent_matrix_decomposition_uses_metal_selected_device") else {
        return;
    };

    let matrix = device.upload(&[4.0_f32, 1.0, 1.0, 3.0]).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let operand = StridedOperand {
        buffer: &matrix,
        layout: &layout,
    };
    let lu = operand.lu(&device).unwrap();
    assert!((lu.det() - 11.0).abs() <= 8.0 * f32::EPSILON);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_match_ordinary_contracts() {
    let Some(device) = device("blocked_pivoted_decompositions_match_ordinary_contracts") else {
        return;
    };

    let square = device.upload(&[2.0f32, 5.0, 1.0, 2.0]).unwrap();
    let square_layout = Layout::c_contiguous([2, 2]).unwrap();
    let square_operand = StridedOperand {
        buffer: &square,
        layout: &square_layout,
    };
    let ordinary_lu = full_piv_lu(&device, square_operand).unwrap();
    let blocked_lu = full_piv_lu_blocked(&device, square_operand).unwrap();
    assert_eq!(blocked_lu.rank(), ordinary_lu.rank());
    assert_eq!(blocked_lu.row_permutation(), ordinary_lu.row_permutation());
    assert_eq!(blocked_lu.col_permutation(), ordinary_lu.col_permutation());
    assert!((blocked_lu.det() - ordinary_lu.det()).abs() <= 1.0e-5);

    let tall = device.upload(&[1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0]).unwrap();
    let tall_layout = Layout::c_contiguous([3, 2]).unwrap();
    let tall_operand = StridedOperand {
        buffer: &tall,
        layout: &tall_layout,
    };
    let ordinary_qr = col_piv_qr(&device, tall_operand).unwrap();
    let blocked_qr = col_piv_qr_blocked(&device, tall_operand).unwrap();
    assert_eq!(blocked_qr.rank(), ordinary_qr.rank());
    assert_eq!(blocked_qr.permutation(), ordinary_qr.permutation());
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_reject_non_dense_operands() {
    let Some(device) = device("blocked_pivoted_decompositions_reject_non_dense_operands") else {
        return;
    };
    let dense = device.upload(&[1.0f32; 16]).unwrap();
    let small = device.upload(&[1.0f32; 4]).unwrap();
    let transposed = Layout::try_new([4, 4], [1, 4], 0).expect("valid test layout");
    let broadcast = Layout::try_new([4, 4], [0, 1], 0).expect("valid test layout");

    let lu_result = full_piv_lu_blocked(
        &device,
        StridedOperand {
            buffer: &dense,
            layout: &transposed,
        },
    );
    assert!(
        matches!(
            lu_result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("dense C-contiguous")
        ),
        "complete-pivoted LU must reject non-dense operands"
    );

    let qr_result = col_piv_qr_blocked(
        &device,
        StridedOperand {
            buffer: &small,
            layout: &broadcast,
        },
    );
    assert!(
        matches!(
            qr_result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("dense C-contiguous")
        ),
        "column-pivoted QR must reject non-dense operands"
    );
}
