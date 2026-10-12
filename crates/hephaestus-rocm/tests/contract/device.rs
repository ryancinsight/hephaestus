use super::*;

#[test]
fn backend_name_is_stable() {
    #[cfg(not(feature = "rocm"))]
    assert_eq!(
        RocmDevice::try_default().unwrap_err().to_string(),
        "no compatible accelerator adapter available: ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
    );

    #[cfg(feature = "rocm")]
    if let Some(device) = device("backend_name_is_stable") {
        assert_eq!(device.backend_name(), "rocm");
    }
}

#[test]
fn device_capabilities_and_topology_are_driver_backed() {
    let Some(device) = device("device_capabilities_and_topology_are_driver_backed") else {
        return;
    };

    let limits = device.device_limits();
    assert!(limits.max_buffer_size > 0);
    assert!(limits.max_compute_workgroup_size_x > 0);
    assert!(limits.max_compute_workgroup_size_y > 0);
    assert!(limits.max_compute_workgroup_size_z > 0);
    assert!(limits.max_compute_invocations_per_workgroup > 0);
    assert!(limits.max_compute_workgroup_storage_size > 0);
    assert_eq!(limits.max_storage_buffers_per_shader_stage, None);
    assert_eq!(
        limits.max_buffers_and_acceleration_structures_per_shader_stage,
        None
    );
    assert_eq!(limits.max_immediate_size, 0);

    assert!(!device.supports_device_feature(DeviceFeature::TimestampQuery));
    assert!(!device.supports_device_feature(DeviceFeature::ShaderF64));
    assert!(!device.supports_device_feature(DeviceFeature::ShaderF16));
    assert!(!device.supports_device_feature(DeviceFeature::ImmediateData));

    let topology = device
        .topology()
        .expect("an acquired ROCm device must have a topology snapshot");
    assert!(topology.compute_units().is_some());
    assert!(topology.warp_width().is_some());
    assert!(topology.max_threads_per_unit().is_some());
    assert!(topology.registers_per_unit().is_some());
    assert!(topology.shared_mem_per_unit_bytes().is_some());
    assert!(topology.memory_bytes().is_some());
    assert_eq!(topology.memory_tier(), themis::MemoryTier::Device);
    assert!(topology.max_resident_warps().is_some_and(|warps| warps > 0));
}

#[test]
fn alloc_zeroed_produces_zero_values() {
    let Some(device) = device("alloc_zeroed_produces_zero_values") else {
        return;
    };

    let buffer = device.alloc_zeroed::<u32>(17).expect("HIP allocation");
    let mut output = [9_u32; 17];
    device.download(&buffer, &mut output).expect("HIP download");
    assert_eq!(output, [0_u32; 17]);
}

#[cfg(feature = "decomposition")]
#[test]
fn bidiagonalization_matches_leto_factors_and_validates_shapes() {
    let Some(device) = device("bidiagonalization_matches_leto_factors_and_validates_shapes") else {
        return;
    };

    let matrix_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device
        .upload(&matrix_values)
        .expect("HIP bidiagonal upload");
    let layout = Layout::c_contiguous([3, 2]).expect("bidiagonal layout");
    let factor = bidiagonalize(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm bidiagonalization");
    let expected = leto_ops::bidiagonalize(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU bidiagonal reference");

    assert_eq!(factor.shape(), (3, 2));
    let expected_u_matrix = expected.u();
    let expected_u = leto::Storage::as_slice(expected_u_matrix.storage());
    let expected_b = leto::Storage::as_slice(expected.b().storage());
    let expected_v = leto::Storage::as_slice(expected.v().storage());
    let mut actual_u = vec![0.0_f32; expected_u.len()];
    let mut actual_b = vec![0.0_f32; expected_b.len()];
    let mut actual_v = vec![0.0_f32; expected_v.len()];
    device
        .download(factor.u_buffer(), &mut actual_u)
        .expect("HIP U download");
    device
        .download(factor.b_buffer(), &mut actual_b)
        .expect("HIP B download");
    device
        .download(factor.v_buffer(), &mut actual_v)
        .expect("HIP V download");
    for (actual, expected) in actual_u.iter().zip(expected_u) {
        assert_near(*actual, *expected, 2048.0);
    }
    for (actual, expected) in actual_b.iter().zip(expected_b) {
        assert_near(*actual, *expected, 2048.0);
    }
    for (actual, expected) in actual_v.iter().zip(expected_v) {
        assert_near(*actual, *expected, 2048.0);
    }
    let mut reconstructed = vec![0.0_f32; matrix_values.len()];
    for row in 0..3 {
        for col in 0..2 {
            let mut value = 0.0_f32;
            for u_col in 0..3 {
                for b_col in 0..2 {
                    value += actual_u[row * 3 + u_col]
                        * actual_b[u_col * 2 + b_col]
                        * actual_v[col * 2 + b_col];
                }
            }
            reconstructed[row * 2 + col] = value;
        }
    }
    for (actual, expected) in reconstructed.iter().zip(matrix_values) {
        assert_near(*actual, expected, 4096.0);
    }

    let wide = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("HIP wide bidiagonal upload");
    let wide_layout = Layout::c_contiguous([2, 3]).expect("wide bidiagonal layout");
    assert!(matches!(
        bidiagonalize(
            &device,
            StridedOperand {
                buffer: &wide,
                layout: &wide_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("m ≥ n")
    ));
}

#[test]
fn fluent_matrix_traits_match_operation_contracts() {
    let Some(device) = device("fluent_matrix_traits_match_operation_contracts") else {
        return;
    };

    let a_values = [1.0_f32, 2.0, 3.0, 4.0];
    let b_values = [2.0_f32, 0.0, 1.0, 2.0];
    let a = device.upload(&a_values).expect("HIP fluent matrix upload");
    let b = device.upload(&b_values).expect("HIP fluent rhs upload");
    let layout = Layout::c_contiguous([2, 2]).expect("fluent matrix layout");
    let a_operand = StridedOperand {
        buffer: &a,
        layout: &layout,
    };
    let b_operand = StridedOperand {
        buffer: &b,
        layout: &layout,
    };
    let normalized_operand = a_operand.as_operand();
    assert_eq!(normalized_operand.layout.shape(), [2, 2]);

    let product = a_operand
        .matmul(&device, &b_operand)
        .expect("ROCm fluent matmul");
    let mut product_actual = [0.0_f32; 4];
    device
        .download(&product, &mut product_actual)
        .expect("HIP fluent matmul download");
    assert_eq!(product_actual, [8.0, 4.0, 18.0, 8.0]);

    let l1 = a_operand.norm_l1(&device).expect("ROCm fluent L1 norm");
    let mut l1_actual = [0.0_f32; 1];
    device
        .download(&l1, &mut l1_actual)
        .expect("HIP fluent L1 download");
    assert_eq!(l1_actual, [10.0]);

    let trace = a_operand.trace(&device).expect("ROCm fluent trace");
    let mut trace_actual = [0.0_f32; 1];
    device
        .download(&trace, &mut trace_actual)
        .expect("HIP fluent trace download");
    assert_eq!(trace_actual, [5.0]);
    assert_eq!(a_operand.rank(&device).expect("ROCm fluent rank"), 2);

    let determinant = a_operand.det(&device).expect("ROCm fluent determinant");
    let mut determinant_actual = [0.0_f32; 1];
    device
        .download(&determinant, &mut determinant_actual)
        .expect("HIP fluent determinant download");
    assert_eq!(determinant_actual, [-2.0]);

    let power = a_operand
        .matpow(&device, 0)
        .expect("ROCm fluent identity power");
    let mut power_actual = [0.0_f32; 4];
    device
        .download(&power, &mut power_actual)
        .expect("HIP fluent power download");
    assert_eq!(power_actual, [1.0, 0.0, 0.0, 1.0]);

    let pseudoinverse = a_operand.pinv(&device).expect("ROCm fluent pseudoinverse");
    let mut pseudoinverse_actual = [0.0_f32; 4];
    device
        .download(&pseudoinverse, &mut pseudoinverse_actual)
        .expect("HIP fluent pseudoinverse download");
    let expected_pseudoinverse = leto_ops::pinv(&leto::ArrayView::<f32, 2>::new(layout, &a_values))
        .expect("CPU fluent pseudoinverse reference");
    let expected_pseudoinverse_slice = leto::Storage::as_slice(expected_pseudoinverse.storage());
    for (actual, expected) in pseudoinverse_actual
        .iter()
        .zip(expected_pseudoinverse_slice)
    {
        assert_near(*actual, *expected, 4096.0);
    }
}

#[test]
fn dense_vector_ops_match_cpu_reference() {
    let Some(dev) = device("dense_vector_ops_match_cpu_reference") else {
        return;
    };
    let ops = RocmVectorOps::new(&dev).expect("ROCm vector kernel descriptors");
    let len = 257;
    let left_host: Vec<f32> = (0..len)
        .map(|index| (index as f32 - 128.0) * 0.03125)
        .collect();
    let right_host: Vec<f32> = (0..len).map(|index| (index as f32 * 0.017) - 2.0).collect();
    let divisor_host: Vec<f32> = right_host.iter().map(|&value| value.abs() + 7.0).collect();
    let left = dev.upload(&left_host).expect("ROCm left upload");
    let right = dev.upload(&right_host).expect("ROCm right upload");
    let divisor = dev.upload(&divisor_host).expect("ROCm divisor upload");
    let tolerance = 8.0 * f32::EPSILON * len as f32;

    let empty = dev.upload(&[] as &[f32]).expect("ROCm empty upload");
    let empty_output = dev.alloc_zeroed::<f32>(0).expect("ROCm empty allocation");
    ops.copy_vector(&dev, &empty, &empty_output)
        .expect("ROCm empty copy");
    ops.scale_vector(&dev, &empty_output, 2.0)
        .expect("ROCm empty scale");
    ops.axpy(&dev, &empty_output, &empty, 2.0)
        .expect("ROCm empty axpy");
    ops.xpay(&dev, &empty_output, &empty, 2.0)
        .expect("ROCm empty xpay");
    ops.subtract_into(&dev, &empty, &empty, &empty_output)
        .expect("ROCm empty subtraction");
    ops.add_into(&dev, &empty, &empty, &empty_output)
        .expect("ROCm empty addition");
    ops.multiply_into(&dev, &empty, &empty, &empty_output)
        .expect("ROCm empty multiplication");
    ops.divide_into(&dev, &empty, &empty, &empty_output)
        .expect("ROCm empty division");
    let empty_dot = ops.dot(&dev, &empty, &empty).expect("ROCm empty dot");
    let empty_norm = ops.norm_l2(&dev, &empty).expect("ROCm empty norm");
    assert_eq!(empty_dot, 0.0);
    assert_eq!(empty_norm, 0.0);

    let copy = dev.alloc_zeroed::<f32>(len).expect("ROCm copy allocation");
    ops.copy_vector(&dev, &left, &copy)
        .expect("ROCm vector copy");
    let mut copied = vec![0.0; len];
    dev.download(&copy, &mut copied)
        .expect("ROCm copy download");
    assert_vector_close(&copied, &left_host, 0.0);

    let scaled = dev.upload(&left_host).expect("ROCm scale upload");
    ops.scale_vector(&dev, &scaled, -1.75)
        .expect("ROCm vector scale");
    let expected_scale: Vec<f32> = left_host.iter().map(|&value| value * -1.75).collect();
    let mut actual = vec![0.0; len];
    dev.download(&scaled, &mut actual)
        .expect("ROCm scale download");
    assert_vector_close(&actual, &expected_scale, 1.0e-6);

    let axpy = dev.upload(&left_host).expect("ROCm axpy upload");
    ops.axpy(&dev, &axpy, &right, 0.625).expect("ROCm axpy");
    let expected_axpy: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| 0.625_f32.mul_add(right, left))
        .collect();
    dev.download(&axpy, &mut actual)
        .expect("ROCm axpy download");
    assert_vector_close(&actual, &expected_axpy, 1.0e-6);

    let xpay = dev.upload(&left_host).expect("ROCm xpay upload");
    ops.xpay(&dev, &xpay, &right, -0.375).expect("ROCm xpay");
    let expected_xpay: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| (-0.375_f32).mul_add(left, right))
        .collect();
    dev.download(&xpay, &mut actual)
        .expect("ROCm xpay download");
    assert_vector_close(&actual, &expected_xpay, 1.0e-6);

    let difference = dev
        .alloc_zeroed::<f32>(len)
        .expect("ROCm subtraction allocation");
    ops.subtract_into(&dev, &left, &right, &difference)
        .expect("ROCm subtraction");
    let expected_difference: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left - right)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("ROCm subtraction download");
    assert_vector_close(&actual, &expected_difference, 1.0e-6);

    ops.add_into(&dev, &left, &right, &difference)
        .expect("ROCm addition");
    let expected_sum: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left + right)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("ROCm addition download");
    assert_vector_close(&actual, &expected_sum, 1.0e-6);

    ops.multiply_into(&dev, &left, &right, &difference)
        .expect("ROCm multiplication");
    let expected_product: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left * right)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("ROCm multiplication download");
    assert_vector_close(&actual, &expected_product, 1.0e-6);

    ops.divide_into(&dev, &left, &divisor, &difference)
        .expect("ROCm division");
    let expected_quotient: Vec<f32> = left_host
        .iter()
        .zip(&divisor_host)
        .map(|(&left, &divisor)| left / divisor)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("ROCm division download");
    assert_vector_close(&actual, &expected_quotient, 1.0e-6);

    let prepared_dot = ops
        .prepare_dot(&dev, &left, &right)
        .expect("ROCm dot preparation");
    let dot = ops
        .dot_prepared(&dev, &prepared_dot, &left, &right)
        .expect("ROCm prepared dot");
    let expected_dot: f32 = left_host
        .iter()
        .zip(&right_host)
        .map(|(&l, &r)| l * r)
        .sum();
    assert!((dot - expected_dot).abs() <= tolerance * expected_dot.abs().max(1.0));

    let prepared_norm = ops
        .prepare_norm_l2(&dev, &left)
        .expect("ROCm norm preparation");
    let norm = ops
        .norm_l2_prepared(&dev, &prepared_norm, &left)
        .expect("ROCm prepared norm");
    let expected_norm = left_host
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - expected_norm).abs() <= tolerance * expected_norm.abs().max(1.0));

    let replacement_host: Vec<f32> = right_host.iter().map(|&value| value * 0.5).collect();
    let replacement = dev
        .upload(&replacement_host)
        .expect("ROCm replacement upload");
    ops.copy_vector(&dev, &replacement, &left)
        .expect("ROCm prepared-input update");
    let updated_dot = ops
        .dot_prepared(&dev, &prepared_dot, &left, &right)
        .expect("ROCm repeated prepared dot");
    let expected_updated_dot: f32 = replacement_host
        .iter()
        .zip(&right_host)
        .map(|(&l, &r)| l * r)
        .sum();
    assert!(
        (updated_dot - expected_updated_dot).abs()
            <= tolerance * expected_updated_dot.abs().max(1.0)
    );

    let short = dev.upload(&vec![0.0; len - 1]).expect("ROCm short upload");
    assert!(matches!(
        ops.axpy(&dev, &left, &short, 1.0),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        ops.add_into(&dev, &left, &short, &difference),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        ops.multiply_into(&dev, &left, &short, &difference),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        ops.divide_into(&dev, &left, &short, &difference),
        Err(HephaestusError::LengthMismatch { .. })
    ));
}
