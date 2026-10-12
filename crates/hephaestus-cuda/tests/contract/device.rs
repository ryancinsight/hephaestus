use super::*;

#[test]
fn device_capabilities_are_driver_backed() {
    let Some(dev) = device("device_capabilities_are_driver_backed") else {
        return;
    };

    let limits = dev.device_limits();
    assert!(limits.max_buffer_size > 0);
    let free = dev.free_memory_bytes().expect("free-memory query");
    assert!(
        free <= limits.max_buffer_size,
        "free memory {free} exceeds the device capacity {}",
        limits.max_buffer_size
    );
    assert!(limits.max_compute_workgroup_size_x > 0);
    assert!(limits.max_compute_workgroup_size_y > 0);
    assert!(limits.max_compute_workgroup_size_z > 0);
    assert!(limits.max_compute_invocations_per_workgroup > 0);
    assert!(limits.max_compute_workgroup_storage_size > 0);
    assert_eq!(limits.max_storage_buffers_per_shader_stage, None);
    assert_eq!(limits.max_immediate_size, 0);

    assert!(dev.supports_device_feature(DeviceFeature::ImmediateData));
    assert!(!dev.supports_device_feature(DeviceFeature::TimestampQuery));
    assert!(!dev.supports_device_feature(DeviceFeature::ShaderF16));
    assert!(!dev.supports_device_feature(DeviceFeature::MappablePrimaryBuffers));
}

#[test]
fn alloc_zeroed_is_zero() {
    let Some(dev) = device("alloc_zeroed_is_zero") else {
        return;
    };
    let buf = dev.alloc_zeroed::<i32>(8).expect("alloc_zeroed");
    assert_eq!(buf.len(), 8);
    let mut out = vec![7i32; 8];
    dev.download(&buf, &mut out).expect("download");
    assert_eq!(out, vec![0i32; 8], "alloc_zeroed must yield zeros");
}

#[test]
fn dense_vector_ops_match_cpu_reference() {
    let Some(dev) = device("dense_vector_ops_match_cpu_reference") else {
        return;
    };
    let ops = CudaVectorOps::new(&dev).expect("CUDA vector kernel descriptors");
    let len = 257;
    let left_host: Vec<f32> = (0..len)
        .map(|index| (index as f32 - 128.0) * 0.03125)
        .collect();
    let right_host: Vec<f32> = (0..len).map(|index| (index as f32 * 0.017) - 2.0).collect();
    let divisor_host: Vec<f32> = right_host.iter().map(|&value| value.abs() + 7.0).collect();
    let left = dev.upload(&left_host).expect("CUDA left upload");
    let right = dev.upload(&right_host).expect("CUDA right upload");
    let divisor = dev.upload(&divisor_host).expect("CUDA divisor upload");
    let tolerance = 8.0 * f32::EPSILON * len as f32;

    let empty = dev.upload(&[] as &[f32]).expect("CUDA empty upload");
    let empty_output = dev.alloc_zeroed::<f32>(0).expect("CUDA empty allocation");
    ops.copy_vector(&dev, &empty, &empty_output)
        .expect("CUDA empty copy");
    ops.scale_vector(&dev, &empty_output, 2.0)
        .expect("CUDA empty scale");
    ops.axpy(&dev, &empty_output, &empty, 2.0)
        .expect("CUDA empty axpy");
    ops.xpay(&dev, &empty_output, &empty, 2.0)
        .expect("CUDA empty xpay");
    ops.subtract_into(&dev, &empty, &empty, &empty_output)
        .expect("CUDA empty subtraction");
    ops.add_into(&dev, &empty, &empty, &empty_output)
        .expect("CUDA empty addition");
    ops.multiply_into(&dev, &empty, &empty, &empty_output)
        .expect("CUDA empty multiplication");
    ops.divide_into(&dev, &empty, &empty, &empty_output)
        .expect("CUDA empty division");
    let empty_dot = ops.dot(&dev, &empty, &empty).expect("CUDA empty dot");
    let empty_norm = ops.norm_l2(&dev, &empty).expect("CUDA empty norm");
    assert_eq!(empty_dot, 0.0);
    assert_eq!(empty_norm, 0.0);

    let copy = dev.alloc_zeroed::<f32>(len).expect("CUDA copy allocation");
    ops.copy_vector(&dev, &left, &copy)
        .expect("CUDA vector copy");
    let mut copied = vec![0.0; len];
    dev.download(&copy, &mut copied)
        .expect("CUDA copy download");
    assert_vector_close(&copied, &left_host, 0.0);

    let scaled = dev.upload(&left_host).expect("CUDA scale upload");
    ops.scale_vector(&dev, &scaled, -1.75)
        .expect("CUDA vector scale");
    let expected_scale: Vec<f32> = left_host.iter().map(|&value| value * -1.75).collect();
    let mut actual = vec![0.0; len];
    dev.download(&scaled, &mut actual)
        .expect("CUDA scale download");
    assert_vector_close(&actual, &expected_scale, 1.0e-6);

    let axpy = dev.upload(&left_host).expect("CUDA axpy upload");
    ops.axpy(&dev, &axpy, &right, 0.625).expect("CUDA axpy");
    let expected_axpy: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| 0.625_f32.mul_add(right, left))
        .collect();
    dev.download(&axpy, &mut actual)
        .expect("CUDA axpy download");
    assert_vector_close(&actual, &expected_axpy, 1.0e-6);

    let xpay = dev.upload(&left_host).expect("CUDA xpay upload");
    ops.xpay(&dev, &xpay, &right, -0.375).expect("CUDA xpay");
    let expected_xpay: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| (-0.375_f32).mul_add(left, right))
        .collect();
    dev.download(&xpay, &mut actual)
        .expect("CUDA xpay download");
    assert_vector_close(&actual, &expected_xpay, 1.0e-6);

    let difference = dev
        .alloc_zeroed::<f32>(len)
        .expect("CUDA subtraction allocation");
    ops.subtract_into(&dev, &left, &right, &difference)
        .expect("CUDA subtraction");
    let expected_difference: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left - right)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("CUDA subtraction download");
    assert_vector_close(&actual, &expected_difference, 1.0e-6);

    ops.add_into(&dev, &left, &right, &difference)
        .expect("CUDA addition");
    let expected_sum: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left + right)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("CUDA addition download");
    assert_vector_close(&actual, &expected_sum, 1.0e-6);

    ops.multiply_into(&dev, &left, &right, &difference)
        .expect("CUDA multiplication");
    let expected_product: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left * right)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("CUDA multiplication download");
    assert_vector_close(&actual, &expected_product, 1.0e-6);

    ops.divide_into(&dev, &left, &divisor, &difference)
        .expect("CUDA division");
    let expected_quotient: Vec<f32> = left_host
        .iter()
        .zip(&divisor_host)
        .map(|(&left, &divisor)| left / divisor)
        .collect();
    dev.download(&difference, &mut actual)
        .expect("CUDA division download");
    assert_vector_close(&actual, &expected_quotient, 1.0e-6);

    let prepared_dot = ops
        .prepare_dot(&dev, &left, &right)
        .expect("CUDA dot preparation");
    let dot = ops
        .dot_prepared(&dev, &prepared_dot, &left, &right)
        .expect("CUDA prepared dot");
    let expected_dot: f32 = left_host
        .iter()
        .zip(&right_host)
        .map(|(&l, &r)| l * r)
        .sum();
    assert!((dot - expected_dot).abs() <= tolerance * expected_dot.abs().max(1.0));

    let prepared_norm = ops
        .prepare_norm_l2(&dev, &left)
        .expect("CUDA norm preparation");
    let norm = ops
        .norm_l2_prepared(&dev, &prepared_norm, &left)
        .expect("CUDA prepared norm");
    let expected_norm = left_host
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - expected_norm).abs() <= tolerance * expected_norm.abs().max(1.0));

    let replacement_host: Vec<f32> = right_host.iter().map(|&value| value * 0.5).collect();
    let replacement = dev
        .upload(&replacement_host)
        .expect("CUDA replacement upload");
    ops.copy_vector(&dev, &replacement, &left)
        .expect("CUDA prepared-input update");
    let updated_dot = ops
        .dot_prepared(&dev, &prepared_dot, &left, &right)
        .expect("CUDA repeated prepared dot");
    let expected_updated_dot: f32 = replacement_host
        .iter()
        .zip(&right_host)
        .map(|(&l, &r)| l * r)
        .sum();
    assert!(
        (updated_dot - expected_updated_dot).abs()
            <= tolerance * expected_updated_dot.abs().max(1.0)
    );

    let short = dev.upload(&vec![0.0; len - 1]).expect("CUDA short upload");
    assert_length_mismatch(ops.axpy(&dev, &left, &short, 1.0), len, len - 1);
    assert_length_mismatch(ops.add_into(&dev, &left, &short, &difference), len, len - 1);
    assert_length_mismatch(
        ops.multiply_into(&dev, &left, &short, &difference),
        len,
        len - 1,
    );
    assert_length_mismatch(
        ops.divide_into(&dev, &left, &short, &difference),
        len,
        len - 1,
    );
}
