use super::*;

#[test]
fn matmul_kernel_matches_cpu_values_across_tile_boundaries_and_rejects_invalid_contracts() {
    let Some(device) = device(
        "matmul_kernel_matches_cpu_values_across_tile_boundaries_and_rejects_invalid_contracts",
    ) else {
        return;
    };

    let lhs_values: Vec<i32> = (0..17 * 19).map(|index| (index % 7) - 3).collect();
    let rhs_values: Vec<i32> = (0..19 * 5).map(|index| (index % 5) - 2).collect();
    let lhs_buffer = device.upload(&lhs_values).expect("HIP matmul lhs upload");
    let rhs_buffer = device.upload(&rhs_values).expect("HIP matmul rhs upload");
    let lhs_layout = Layout::c_contiguous([17, 19]).expect("matmul lhs layout");
    let rhs_layout = Layout::c_contiguous([19, 5]).expect("matmul rhs layout");
    let lhs = StridedOperand {
        buffer: &lhs_buffer,
        layout: &lhs_layout,
    };
    let rhs = StridedOperand {
        buffer: &rhs_buffer,
        layout: &rhs_layout,
    };

    let output = matmul(&device, lhs, rhs).expect("HIP tiled matmul");
    let mut output_values = vec![0_i32; 17 * 5];
    device
        .download(&output, &mut output_values)
        .expect("HIP matmul download");
    let mut expected: Vec<i32> = Vec::with_capacity(17 * 5);
    for row in 0..17 {
        for col in 0..5 {
            expected.push(
                (0..19)
                    .map(|shared| lhs_values[row * 19 + shared] * rhs_values[shared * 5 + col])
                    .sum(),
            );
        }
    }
    assert_eq!(output_values, expected);

    let output_layout = Layout::c_contiguous([17, 5]).expect("matmul output layout");
    let output_into = device
        .alloc_zeroed::<i32>(17 * 5)
        .expect("HIP caller-owned matmul output");
    matmul_into(
        &device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &output_into,
            layout: &output_layout,
        },
    )
    .expect("HIP caller-owned tiled matmul");
    let mut output_into_values = vec![0_i32; 17 * 5];
    device
        .download(&output_into, &mut output_into_values)
        .expect("HIP caller-owned matmul download");
    assert_eq!(output_into_values, expected);

    let wrong_layout = Layout::c_contiguous([17, 4]).expect("wrong matmul output layout");
    let wrong_output = device
        .alloc_zeroed::<i32>(17 * 4)
        .expect("wrong matmul output buffer");
    assert!(matches!(
        matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &wrong_output,
                layout: &wrong_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("matmul dimension mismatch")
    ));

    assert!(matches!(
        matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &lhs_buffer,
                layout: &lhs_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output buffer must not alias either input buffer"
    ));
}

#[test]
fn batched_matmul_kernel_matches_cpu_values_with_broadcast_and_rejects_invalid_contracts() {
    let Some(device) = device(
        "batched_matmul_kernel_matches_cpu_values_with_broadcast_and_rejects_invalid_contracts",
    ) else {
        return;
    };

    let lhs_values: Vec<i32> = (0..3 * 17 * 19).map(|index| (index % 7) - 3).collect();
    let rhs_values: Vec<i32> = (0..19 * 5).map(|index| (index % 5) - 2).collect();
    let lhs_buffer = device.upload(&lhs_values).expect("HIP batched lhs upload");
    let rhs_buffer = device
        .upload(&rhs_values)
        .expect("HIP broadcast rhs upload");
    let lhs_layout = Layout::c_contiguous([3, 17, 19]).expect("batched lhs layout");
    let rhs_layout = Layout::c_contiguous([1, 19, 5]).expect("broadcast rhs layout");
    let lhs = StridedOperand {
        buffer: &lhs_buffer,
        layout: &lhs_layout,
    };
    let rhs = StridedOperand {
        buffer: &rhs_buffer,
        layout: &rhs_layout,
    };

    let output = batched_matmul(&device, lhs, rhs).expect("HIP batched matmul");
    let mut output_values = vec![0_i32; 3 * 17 * 5];
    device
        .download(&output, &mut output_values)
        .expect("HIP batched matmul download");
    let mut expected: Vec<i32> = Vec::with_capacity(3 * 17 * 5);
    for batch in 0..3 {
        for row in 0..17 {
            for col in 0..5 {
                expected.push(
                    (0..19)
                        .map(|shared| {
                            lhs_values[(batch * 17 + row) * 19 + shared]
                                * rhs_values[shared * 5 + col]
                        })
                        .sum(),
                );
            }
        }
    }
    assert_eq!(output_values, expected);

    let output_layout = Layout::c_contiguous([3, 17, 5]).expect("batched output layout");
    let output_into = device
        .alloc_zeroed::<i32>(3 * 17 * 5)
        .expect("HIP caller-owned batched output");
    batched_matmul_into(
        &device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &output_into,
            layout: &output_layout,
        },
    )
    .expect("HIP caller-owned batched matmul");
    let mut output_into_values = vec![0_i32; 3 * 17 * 5];
    device
        .download(&output_into, &mut output_into_values)
        .expect("HIP caller-owned batched download");
    assert_eq!(output_into_values, expected);

    let wrong_layout = Layout::c_contiguous([2, 17, 5]).expect("wrong batched output layout");
    let wrong_output = device
        .alloc_zeroed::<i32>(2 * 17 * 5)
        .expect("wrong batched output buffer");
    assert!(matches!(
        batched_matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &wrong_output,
                layout: &wrong_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("batched matmul shape mismatch")
    ));

    assert!(matches!(
        batched_matmul_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &lhs_buffer,
                layout: &lhs_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output buffer must not alias either input buffer"
    ));
}

#[test]
fn kron_kernel_matches_cpu_values_for_strided_operands_and_rejects_invalid_contracts() {
    let Some(device) =
        device("kron_kernel_matches_cpu_values_for_strided_operands_and_rejects_invalid_contracts")
    else {
        return;
    };

    let lhs_values = [99_i32, 1, 2, 3, 4];
    let rhs_values = [5_i32, 6, 7, 8];
    let lhs_buffer = device.upload(&lhs_values).expect("HIP kron lhs upload");
    let rhs_buffer = device.upload(&rhs_values).expect("HIP kron rhs upload");
    let lhs_layout = Layout::try_new([2, 2], [2, 1], 1).expect("valid test layout");
    let rhs_layout = Layout::c_contiguous([2, 2]).expect("kron rhs layout");
    let lhs = StridedOperand {
        buffer: &lhs_buffer,
        layout: &lhs_layout,
    };
    let rhs = StridedOperand {
        buffer: &rhs_buffer,
        layout: &rhs_layout,
    };
    let expected = vec![
        5_i32, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32,
    ];

    let output = kron(&device, lhs, rhs).expect("HIP kron");
    let mut output_values = vec![0_i32; 16];
    device
        .download(&output, &mut output_values)
        .expect("HIP kron download");
    assert_eq!(output_values, expected);

    let strided_output_layout = Layout::try_new([4, 4], [8, 2], 0).expect("valid test layout");
    let strided_output = device
        .alloc_zeroed::<i32>(31)
        .expect("HIP strided kron output");
    kron_into(
        &device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &strided_output,
            layout: &strided_output_layout,
        },
    )
    .expect("HIP caller-owned strided kron");
    let mut strided_values = vec![0_i32; 31];
    device
        .download(&strided_output, &mut strided_values)
        .expect("HIP strided kron download");
    let gathered: Vec<_> = strided_values
        .chunks_exact(8)
        .take(4)
        .flat_map(|row| row.iter().step_by(2).take(4).copied())
        .collect();
    assert_eq!(gathered, expected);

    let empty_lhs_buffer = device.upload(&[]).expect("HIP empty kron lhs upload");
    let empty_lhs_layout = Layout::c_contiguous([0, 2]).expect("empty kron lhs layout");
    let empty_output = kron(
        &device,
        StridedOperand {
            buffer: &empty_lhs_buffer,
            layout: &empty_lhs_layout,
        },
        rhs,
    )
    .expect("HIP empty kron");
    assert_eq!(empty_output.len(), 0);

    let wrong_layout = Layout::c_contiguous([3, 4]).expect("wrong kron output layout");
    let wrong_output = device
        .alloc_zeroed::<i32>(12)
        .expect("wrong kron output buffer");
    assert!(matches!(
        kron_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &wrong_output,
                layout: &wrong_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("Kronecker output shape mismatch")
    ));

    assert!(matches!(
        kron_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &lhs_buffer,
                layout: &lhs_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "output buffer must not alias either input buffer"
    ));

    let aliased_output_layout = Layout::try_new([4, 4], [0, 1], 0).expect("valid test layout");
    let aliased_output = device
        .alloc_zeroed::<i32>(4)
        .expect("zero-stride kron output buffer");
    assert!(matches!(
        kron_into(
            &device,
            lhs,
            rhs,
            StridedOperand {
                buffer: &aliased_output,
                layout: &aliased_output_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "Kronecker output layout must not contain zero-stride aliasing"
    ));
}

#[test]
fn matpow_matches_cpu_values_for_strided_inputs_and_rejects_non_square() {
    let Some(device) =
        device("matpow_matches_cpu_values_for_strided_inputs_and_rejects_non_square")
    else {
        return;
    };

    let shear_values = [1_i32, 1, 0, 1];
    let shear_buffer = device
        .upload(&shear_values)
        .expect("HIP matpow shear upload");
    let square_layout = Layout::c_contiguous([2, 2]).expect("matpow square layout");
    let shear = matpow(
        &device,
        StridedOperand {
            buffer: &shear_buffer,
            layout: &square_layout,
        },
        5,
    )
    .expect("HIP matpow shear");
    let mut shear_output = [0_i32; 4];
    device
        .download(&shear, &mut shear_output)
        .expect("HIP matpow shear download");
    assert_eq!(shear_output, [1, 5, 0, 1]);

    let strided_values = [99_i32, 1, 2, 3, 4];
    let strided_buffer = device
        .upload(&strided_values)
        .expect("HIP strided matpow upload");
    let strided_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid test layout");
    let strided_power = matpow(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
        2,
    )
    .expect("HIP strided matpow");
    let mut strided_output = [0_i32; 4];
    device
        .download(&strided_power, &mut strided_output)
        .expect("HIP strided matpow download");
    assert_eq!(strided_output, [7, 15, 10, 22]);

    let identity_power = matpow(
        &device,
        StridedOperand {
            buffer: &strided_buffer,
            layout: &strided_layout,
        },
        0,
    )
    .expect("HIP matpow identity");
    let mut identity_output = [0_i32; 4];
    device
        .download(&identity_power, &mut identity_output)
        .expect("HIP matpow identity download");
    assert_eq!(identity_output, [1, 0, 0, 1]);

    let alternate_buffer = device
        .upload(&[AlternateIdentity(4); 4])
        .expect("HIP alternate identity upload");
    let alternate_power = matpow(
        &device,
        StridedOperand {
            buffer: &alternate_buffer,
            layout: &square_layout,
        },
        0,
    )
    .expect("HIP alternate identity matpow");
    let mut alternate_output = [AlternateIdentity(0); 4];
    device
        .download(&alternate_power, &mut alternate_output)
        .expect("HIP alternate identity download");
    assert_eq!(
        alternate_output,
        [
            AlternateIdentity(9),
            AlternateIdentity(-7),
            AlternateIdentity(-7),
            AlternateIdentity(9),
        ]
    );

    let empty_buffer = device.upload::<i32>(&[]).expect("HIP empty matpow upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty matpow layout");
    let empty_power = matpow(
        &device,
        StridedOperand {
            buffer: &empty_buffer,
            layout: &empty_layout,
        },
        0,
    )
    .expect("HIP empty matpow");
    assert_eq!(empty_power.len(), 0);

    let nonsquare_values = [1_i32, 2, 3, 4, 5, 6];
    let nonsquare_buffer = device
        .upload(&nonsquare_values)
        .expect("HIP nonsquare matpow upload");
    let nonsquare_layout = Layout::c_contiguous([2, 3]).expect("nonsquare matpow layout");
    assert!(matches!(
        matpow(
            &device,
            StridedOperand {
                buffer: &nonsquare_buffer,
                layout: &nonsquare_layout,
            },
            2,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "matpow requires a square matrix, got shape [2, 3]"
    ));
}

#[test]
fn seeded_random_initializers_match_determinism_and_distribution_contracts() {
    let Some(device) =
        device("seeded_random_initializers_match_determinism_and_distribution_contracts")
    else {
        return;
    };

    let shape = [1000];
    let uniform =
        uniform_with_seed(&device, shape, -2.0_f32, 5.0, 42).expect("HIP uniform initializer");
    let mut uniform_values = vec![0.0_f32; 1000];
    device
        .download(&uniform, &mut uniform_values)
        .expect("HIP uniform download");
    let uniform_again = uniform_with_seed(&device, shape, -2.0_f32, 5.0, 42)
        .expect("HIP repeated uniform initializer");
    let mut uniform_again_values = vec![0.0_f32; 1000];
    device
        .download(&uniform_again, &mut uniform_again_values)
        .expect("HIP repeated uniform download");
    assert_eq!(uniform_values, uniform_again_values);
    assert!(
        uniform_values
            .iter()
            .all(|&value| (-2.0..5.0).contains(&value))
    );

    let normal =
        normal_with_seed(&device, shape, 0.0_f32, 1.0, 42).expect("HIP normal initializer");
    let mut normal_values = vec![0.0_f32; 1000];
    device
        .download(&normal, &mut normal_values)
        .expect("HIP normal download");
    assert!(normal_values.iter().any(|&value| value != 0.0));
}
