//! Differential contract tests for Metal dense vector operations.

use hephaestus_core::{ComputeDevice, DenseVectorOps, DeviceBuffer, HephaestusError};
use hephaestus_metal::{MetalBuffer, MetalDevice, MetalVectorOps};

fn device_or_skip() -> Option<MetalDevice> {
    static DEVICE: std::sync::OnceLock<Option<MetalDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(|| match MetalDevice::try_default() {
            Ok(device) => Some(device),
            Err(error) => {
                if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some() {
                    panic!("Metal device required for dense-vector contract: {error}");
                }
                eprintln!("skipping Metal dense-vector test: {error}");
                None
            }
        })
        .clone()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * want.abs().max(1.0),
            "index {index}: got {got}, expected {want}"
        );
    }
}

fn read_back(device: &MetalDevice, buffer: &MetalBuffer<f32>) -> Vec<f32> {
    let mut host = vec![0.0; buffer.len()];
    device
        .download(buffer, &mut host)
        .expect("invariant: Metal readback succeeds");
    host
}

#[test]
fn dense_vector_ops_match_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = MetalVectorOps::new(&device).expect("invariant: vector kernels compile");
    let len = 257;
    let left_host: Vec<f32> = (0..len)
        .map(|index| (index as f32 - 128.0) * 0.03125)
        .collect();
    let right_host: Vec<f32> = (0..len).map(|index| index as f32 * 0.017 - 2.0).collect();
    let left = device.upload(&left_host).expect("Metal left upload");
    let right = device.upload(&right_host).expect("Metal right upload");
    let tolerance = 8.0 * f32::EPSILON * len as f32;

    let empty = device.upload(&[] as &[f32]).expect("Metal empty upload");
    let empty_output = device
        .alloc_zeroed::<f32>(0)
        .expect("Metal empty allocation");
    ops.copy_vector(&device, &empty, &empty_output)
        .expect("Metal empty copy");
    ops.scale_vector(&device, &empty_output, 2.0)
        .expect("Metal empty scale");
    ops.axpy(&device, &empty_output, &empty, 2.0)
        .expect("Metal empty axpy");
    ops.xpay(&device, &empty_output, &empty, 2.0)
        .expect("Metal empty xpay");
    ops.subtract_into(&device, &empty, &empty, &empty_output)
        .expect("Metal empty subtraction");
    assert_eq!(
        ops.dot(&device, &empty, &empty).expect("Metal empty dot"),
        0.0
    );
    assert_eq!(ops.norm_l2(&device, &empty).expect("Metal empty norm"), 0.0);

    let copy = device
        .alloc_zeroed::<f32>(len)
        .expect("Metal copy allocation");
    ops.copy_vector(&device, &left, &copy)
        .expect("Metal vector copy");
    assert_close(&read_back(&device, &copy), &left_host, 0.0);

    let scaled = device.upload(&left_host).expect("Metal scale upload");
    ops.scale_vector(&device, &scaled, -1.75)
        .expect("Metal vector scale");
    let expected_scale: Vec<f32> = left_host.iter().map(|&value| value * -1.75).collect();
    assert_close(&read_back(&device, &scaled), &expected_scale, 1.0e-6);

    let axpy = device.upload(&left_host).expect("Metal axpy upload");
    ops.axpy(&device, &axpy, &right, 0.625).expect("Metal axpy");
    let expected_axpy: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| 0.625_f32.mul_add(right, left))
        .collect();
    assert_close(&read_back(&device, &axpy), &expected_axpy, 1.0e-6);

    let xpay = device.upload(&left_host).expect("Metal xpay upload");
    ops.xpay(&device, &xpay, &right, -0.375)
        .expect("Metal xpay");
    let expected_xpay: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| (-0.375_f32).mul_add(left, right))
        .collect();
    assert_close(&read_back(&device, &xpay), &expected_xpay, 1.0e-6);

    let difference = device
        .alloc_zeroed::<f32>(len)
        .expect("Metal subtraction allocation");
    ops.subtract_into(&device, &left, &right, &difference)
        .expect("Metal subtraction");
    let expected_difference: Vec<f32> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left - right)
        .collect();
    assert_close(
        &read_back(&device, &difference),
        &expected_difference,
        1.0e-6,
    );

    let prepared_dot = ops
        .prepare_dot(&device, &left, &right)
        .expect("Metal dot preparation");
    let dot = ops
        .dot_prepared(&device, &prepared_dot, &left, &right)
        .expect("Metal prepared dot");
    let expected_dot: f32 = left_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left * right)
        .sum();
    assert!((dot - expected_dot).abs() <= tolerance * expected_dot.abs().max(1.0));

    let prepared_norm = ops
        .prepare_norm_l2(&device, &left)
        .expect("Metal norm preparation");
    let norm = ops
        .norm_l2_prepared(&device, &prepared_norm, &left)
        .expect("Metal prepared norm");
    let expected_norm = left_host
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - expected_norm).abs() <= tolerance * expected_norm.abs().max(1.0));

    let other = device.upload(&left_host).expect("Metal other upload");
    assert!(matches!(
        ops.dot_prepared(&device, &prepared_dot, &other, &right),
        Err(HephaestusError::DispatchFailed { .. })
    ));
    assert!(matches!(
        ops.norm_l2_prepared(&device, &prepared_norm, &other),
        Err(HephaestusError::DispatchFailed { .. })
    ));

    let replacement_host: Vec<f32> = right_host.iter().map(|&value| value * 0.5).collect();
    let replacement = device
        .upload(&replacement_host)
        .expect("Metal replacement upload");
    ops.copy_vector(&device, &replacement, &left)
        .expect("Metal prepared-input update");
    let updated_dot = ops
        .dot_prepared(&device, &prepared_dot, &left, &right)
        .expect("Metal repeated prepared dot");
    let expected_updated_dot: f32 = replacement_host
        .iter()
        .zip(&right_host)
        .map(|(&left, &right)| left * right)
        .sum();
    assert!(
        (updated_dot - expected_updated_dot).abs()
            <= tolerance * expected_updated_dot.abs().max(1.0)
    );

    let short = device
        .upload(&vec![0.0; len - 1])
        .expect("Metal short upload");
    assert!(matches!(
        ops.axpy(&device, &left, &short, 1.0),
        Err(HephaestusError::LengthMismatch { .. })
    ));
}
