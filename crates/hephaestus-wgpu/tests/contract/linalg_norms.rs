use super::*;

pub(crate) fn linalg_norms_match_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, norm_l1, norm_l2, norm_max};
    use leto::Layout;

    let a_host = vec![-1.0f32, 2.0, -3.0, 4.0];
    let a = device.upload(&a_host).unwrap();
    let a_layout = Layout::c_contiguous([4]).unwrap();
    let operand = StridedOperand {
        buffer: &a,
        layout: &a_layout,
    };

    // L1 norm: 1 + 2 + 3 + 4 = 10
    let l1_buf = norm_l1(&device, operand).unwrap();
    let mut got_l1 = [0.0f32; 1];
    device.download(&l1_buf, &mut got_l1).unwrap();
    assert_eq!(got_l1[0], 10.0);

    // L2 norm: sqrt(1 + 4 + 9 + 16) = sqrt(30)
    let l2_buf = norm_l2(&device, operand).unwrap();
    let mut got_l2 = [0.0f32; 1];
    device.download(&l2_buf, &mut got_l2).unwrap();
    let expected_l2 = 30.0f32.sqrt();
    let l2_tolerance = 2.0 * f32::EPSILON * expected_l2.max(1.0);
    assert!(
        (got_l2[0] - expected_l2).abs() <= l2_tolerance,
        "l2 norm mismatch: got {}, expected {}, tolerance {}",
        got_l2[0],
        expected_l2,
        l2_tolerance
    );

    // Max norm: max(1, 2, 3, 4) = 4
    let max_buf = norm_max(&device, operand).unwrap();
    let mut got_max = [0.0f32; 1];
    device.download(&max_buf, &mut got_max).unwrap();
    assert_eq!(got_max[0], 4.0);
}

#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub(crate) fn linalg_matrix_function_diagnostics_are_stable() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, matexp};
    use leto::Layout;

    let rectangular_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rectangular = device.upload(&rectangular_host).unwrap();
    let rectangular_layout = Layout::c_contiguous([2, 3]).unwrap();
    let rectangular_result = matexp(
        &device,
        StridedOperand {
            buffer: &rectangular,
            layout: &rectangular_layout,
        },
    );
    assert!(matches!(
        rectangular_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Matrix exponential requires square matrix")
    ));

    let non_finite_host = vec![1.0f32, f32::NAN, 0.0, 1.0];
    let non_finite = device.upload(&non_finite_host).unwrap();
    let non_finite_layout = Layout::c_contiguous([2, 2]).unwrap();
    let non_finite_result = matexp(
        &device,
        StridedOperand {
            buffer: &non_finite,
            layout: &non_finite_layout,
        },
    );
    assert!(matches!(
        non_finite_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Matrix exponential failed")
    ));
}

#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub(crate) fn test_wgpu_uniform_and_normal_with_seed() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{normal_with_seed, uniform_with_seed};

    let shape = [1000];
    let low = -2.0f32;
    let high = 5.0f32;
    let u_buf = uniform_with_seed(&device, shape, low, high, 42).unwrap();
    let mut got_u = vec![0.0f32; 1000];
    device.download(&u_buf, &mut got_u).unwrap();

    // Verify determinism & range
    let u_buf_2 = uniform_with_seed(&device, shape, low, high, 42).unwrap();
    let mut got_u_2 = vec![0.0f32; 1000];
    device.download(&u_buf_2, &mut got_u_2).unwrap();
    assert_eq!(got_u, got_u_2);

    for &val in &got_u {
        assert!(val >= low && val < high, "value out of bounds: {val}");
    }

    let n_buf = normal_with_seed(&device, shape, 0.0f32, 1.0f32, 42).unwrap();
    let mut got_n = vec![0.0f32; 1000];
    device.download(&n_buf, &mut got_n).unwrap();
    assert!(got_n.iter().any(|&val| val != 0.0));
}
