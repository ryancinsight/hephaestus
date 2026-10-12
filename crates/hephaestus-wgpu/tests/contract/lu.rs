use super::*;

#[cfg(feature = "decomposition")]
pub(crate) fn full_piv_lu_reconstructs_and_matches_leto_oracles() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, full_piv_lu};
    use leto::Layout;

    let n = 4usize;
    let matrix_host = vec![
        2.0f32, 5.0, -2.0, 2.0, //
        1.0, 2.0, 3.0, 1.0, //
        -2.0, 4.0, 3.0, -2.0, //
        2.0, 1.0, -1.0, -1.0,
    ];
    let rhs_host = vec![3.0f32, -1.0, 2.0, 5.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_lu = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.n(), n);
    assert_eq!(gpu_lu.rank(), n);
    let mut packed = vec![0.0f32; n * n];
    device.download(gpu_lu.lu_buffer(), &mut packed).unwrap();
    let lu = packed_lu_product(&packed, n);
    for row in 0..n {
        for col in 0..n {
            let expected =
                matrix_host[gpu_lu.row_permutation()[row] * n + gpu_lu.col_permutation()[col]];
            assert_close(lu[row * n + col], expected, 1.0e-4);
        }
    }

    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([n], rhs_host).unwrap();
    let leto_lu = leto_ops::full_piv_lu(&leto_matrix.view()).unwrap();
    assert_close(gpu_lu.det(), leto_lu.det(), 1.0e-4);

    let solution = gpu_lu.solve(&device, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; n];
    device.download(&solution, &mut got_solution).unwrap();
    for (actual, expected) in got_solution
        .iter()
        .zip(leto::Storage::as_slice(expected_solution.storage()))
    {
        assert_close(*actual, *expected, 1.0e-4);
    }

    let inverse = gpu_lu.inv(&device).unwrap();
    let expected_inverse = leto_lu.inv().unwrap();
    let mut got_inverse = vec![0.0f32; n * n];
    device.download(&inverse, &mut got_inverse).unwrap();
    for (actual, expected) in got_inverse
        .iter()
        .zip(leto::Storage::as_slice(expected_inverse.storage()))
    {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn full_piv_lu_rejects_rectangular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, full_piv_lu};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );

    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_pivoted_decompositions_match_ordinary_contracts() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{
        StridedOperand, col_piv_qr, col_piv_qr_blocked, full_piv_lu, full_piv_lu_blocked,
    };
    use leto::Layout;

    let square_host = [2.0f32, 5.0, -2.0, 1.0, 2.0, 3.0, -2.0, 4.0, 3.0];
    let square = device.upload(&square_host).unwrap();
    let square_layout = Layout::c_contiguous([3, 3]).unwrap();
    let square_operand = StridedOperand {
        buffer: &square,
        layout: &square_layout,
    };
    let ordinary_lu = full_piv_lu(&device, square_operand).unwrap();
    let blocked_lu = full_piv_lu_blocked(&device, square_operand).unwrap();
    assert_eq!(blocked_lu.rank(), ordinary_lu.rank());
    assert_eq!(blocked_lu.row_permutation(), ordinary_lu.row_permutation());
    assert_eq!(blocked_lu.col_permutation(), ordinary_lu.col_permutation());
    assert_close(blocked_lu.det(), ordinary_lu.det(), 1.0e-5);
    let mut ordinary_factors = vec![0.0f32; 9];
    let mut blocked_factors = vec![0.0f32; 9];
    device
        .download(ordinary_lu.lu_buffer(), &mut ordinary_factors)
        .unwrap();
    device
        .download(blocked_lu.lu_buffer(), &mut blocked_factors)
        .unwrap();
    assert_close_slice(&blocked_factors, &ordinary_factors, 1.0e-5, 0.0);

    let tall_host = [1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];
    let tall = device.upload(&tall_host).unwrap();
    let tall_layout = Layout::c_contiguous([3, 2]).unwrap();
    let tall_operand = StridedOperand {
        buffer: &tall,
        layout: &tall_layout,
    };
    let ordinary_qr = col_piv_qr(&device, tall_operand).unwrap();
    let blocked_qr = col_piv_qr_blocked(&device, tall_operand).unwrap();
    assert_eq!(blocked_qr.rank(), ordinary_qr.rank());
    assert_eq!(blocked_qr.permutation(), ordinary_qr.permutation());
    let mut ordinary_q = vec![0.0f32; 9];
    let mut blocked_q = vec![0.0f32; 9];
    let mut ordinary_r = vec![0.0f32; 6];
    let mut blocked_r = vec![0.0f32; 6];
    device.download(ordinary_qr.q(), &mut ordinary_q).unwrap();
    device.download(blocked_qr.q(), &mut blocked_q).unwrap();
    device.download(ordinary_qr.r(), &mut ordinary_r).unwrap();
    device.download(blocked_qr.r(), &mut blocked_r).unwrap();
    assert_close_slice(&blocked_q, &ordinary_q, 1.0e-5, 0.0);
    assert_close_slice(&blocked_r, &ordinary_r, 1.0e-5, 0.0);
}

/// The blocked-LU differential is one shared clause (SUBSTRATE-003): the
/// fixtures and assertions live in `hephaestus-conformance`, generic over the
/// `BlockedDecompositionBackend` seam the shared `blocked_lu` loop drives.
/// This instantiation supplies the device and the panel width the blocked
/// entry points factor with.
#[cfg(feature = "decomposition")]
pub(crate) fn blocked_lu_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    const BLOCK_SIZE: usize = 64;
    hephaestus_conformance::assert_blocked_lu_contract(&device, BLOCK_SIZE);
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_lu_rejects_non_dense_operands() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_blocked_rejects_non_dense(&device, hephaestus_wgpu::lu_decompose_blocked, "LU");
}

#[cfg(feature = "decomposition")]
pub(crate) fn blocked_pivoted_decompositions_reject_non_dense_operands() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_blocked_rejects_non_dense(
        &device,
        hephaestus_wgpu::full_piv_lu_blocked,
        "complete-pivoted LU",
    );
    assert_blocked_rejects_non_dense(
        &device,
        hephaestus_wgpu::col_piv_qr_blocked,
        "column-pivoted QR",
    );
}
