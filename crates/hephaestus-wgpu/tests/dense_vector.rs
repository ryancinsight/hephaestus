//! Differential contract tests for the dense vector-operation seam: device
//! results against a CPU reference over the same inputs.

use hephaestus_core::{ComputeDevice, DenseVectorOps, DeviceBuffer, HephaestusError};
use hephaestus_wgpu::{WgpuDevice, WgpuVectorOps};

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: std::sync::OnceLock<Option<WgpuDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(
            || match WgpuDevice::try_default("hephaestus-dense-vector-test") {
                Ok(device) => Some(device),
                Err(error) => {
                    eprintln!("skipping wgpu dense-vector test: {error}");
                    None
                }
            },
        )
        .clone()
}

/// Lengths spanning under, exactly, and over one workgroup, plus the empty
/// case, so the tail guard in each kernel is exercised rather than assumed.
const LENGTHS: [usize; 5] = [0, 1, 255, 256, 1000];

fn ascending(len: usize) -> Vec<f32> {
    (0..len).map(|index| 0.5 + index as f32 * 0.25).collect()
}

fn descending(len: usize) -> Vec<f32> {
    (0..len).map(|index| 3.0 - index as f32 * 0.125).collect()
}

fn read_back(device: &WgpuDevice, buffer: &hephaestus_wgpu::WgpuBuffer<f32>) -> Vec<f32> {
    let mut host = vec![0.0_f32; buffer.len()];
    device
        .download(buffer, &mut host)
        .expect("invariant: readback succeeds");
    host
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * want.abs().max(1.0),
            "index {index}: got {got}, want {want}"
        );
    }
}

#[test]
fn axpy_matches_the_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("invariant: kernels compile");

    for len in LENGTHS {
        let target_host = ascending(len);
        let source_host = descending(len);
        let factor = -1.75_f32;

        let target = device.upload(&target_host).expect("invariant: upload");
        let source = device.upload(&source_host).expect("invariant: upload");
        ops.axpy(&device, &target, &source, factor)
            .expect("invariant: matching lengths");

        let expected: Vec<f32> = target_host
            .iter()
            .zip(source_host.iter())
            .map(|(&t, &s)| factor.mul_add(s, t))
            .collect();
        assert_close(&read_back(&device, &target), &expected, 1e-6);
    }
}

#[test]
fn xpay_matches_the_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("invariant: kernels compile");

    for len in LENGTHS {
        let target_host = ascending(len);
        let source_host = descending(len);
        let factor = 0.6_f32;

        let target = device.upload(&target_host).expect("invariant: upload");
        let source = device.upload(&source_host).expect("invariant: upload");
        ops.xpay(&device, &target, &source, factor)
            .expect("invariant: matching lengths");

        let expected: Vec<f32> = target_host
            .iter()
            .zip(source_host.iter())
            .map(|(&t, &s)| factor.mul_add(t, s))
            .collect();
        assert_close(&read_back(&device, &target), &expected, 1e-6);
    }
}

#[test]
fn scale_and_copy_match_the_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("invariant: kernels compile");

    for len in LENGTHS {
        let host = ascending(len);
        let target = device.upload(&host).expect("invariant: upload");
        ops.scale_vector(&device, &target, -2.5)
            .expect("invariant: dispatch");
        let expected: Vec<f32> = host.iter().map(|&value| value * -2.5).collect();
        assert_close(&read_back(&device, &target), &expected, 1e-6);

        let copy = device
            .alloc_zeroed::<f32>(len)
            .expect("invariant: allocation");
        ops.copy_vector(&device, &target, &copy)
            .expect("invariant: matching lengths");
        assert_close(&read_back(&device, &copy), &expected, 0.0);
    }
}

#[test]
fn subtract_matches_the_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("invariant: kernels compile");

    for len in LENGTHS {
        let left_host = ascending(len);
        let right_host = descending(len);
        let left = device.upload(&left_host).expect("invariant: upload");
        let right = device.upload(&right_host).expect("invariant: upload");
        let output = device
            .alloc_zeroed::<f32>(len)
            .expect("invariant: allocation");

        ops.subtract_into(&device, &left, &right, &output)
            .expect("invariant: matching lengths");

        let expected: Vec<f32> = left_host
            .iter()
            .zip(right_host.iter())
            .map(|(&l, &r)| l - r)
            .collect();
        assert_close(&read_back(&device, &output), &expected, 1e-6);
    }
}

#[test]
fn reductions_match_the_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("invariant: kernels compile");

    for len in [1_usize, 255, 256, 1000] {
        let left_host = ascending(len);
        let right_host = descending(len);
        let left = device.upload(&left_host).expect("invariant: upload");
        let right = device.upload(&right_host).expect("invariant: upload");

        // The device reduces in a tree while the reference sums sequentially,
        // so the comparison is epsilon-bounded rather than bitwise, with the
        // bound scaled by the element count that the reordering spans.
        let tolerance = 1e-5 * len as f32;

        let dot = ops
            .dot(&device, &left, &right)
            .expect("invariant: matching lengths");
        let expected_dot: f32 = left_host
            .iter()
            .zip(right_host.iter())
            .map(|(&l, &r)| l * r)
            .sum();
        assert!(
            (dot - expected_dot).abs() <= tolerance * expected_dot.abs().max(1.0),
            "len {len}: dot {dot} vs {expected_dot}"
        );

        let norm = ops
            .norm_l2(&device, &left)
            .expect("invariant: valid operand");
        let expected_norm = left_host.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - expected_norm).abs() <= tolerance * expected_norm.abs().max(1.0),
            "len {len}: norm {norm} vs {expected_norm}"
        );
    }
}

#[test]
fn prepared_reductions_reuse_their_bound_allocation() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("invariant: kernels compile");

    let host = ascending(512);
    let vector = device.upload(&host).expect("invariant: upload");
    let prepared = ops
        .prepare_norm_l2(&device, &vector)
        .expect("invariant: preparation");

    // Repeated execution against the bound allocation is stable.
    let first = ops
        .norm_l2_prepared(&device, &prepared, &vector)
        .expect("invariant: bound operand");
    let second = ops
        .norm_l2_prepared(&device, &prepared, &vector)
        .expect("invariant: bound operand");
    assert!((first - second).abs() <= f32::EPSILON * first.abs().max(1.0));

    // A different allocation is rejected rather than silently reduced.
    let other = device.upload(&host).expect("invariant: upload");
    assert!(matches!(
        ops.norm_l2_prepared(&device, &prepared, &other),
        Err(HephaestusError::DispatchFailed { .. })
    ));
}

#[test]
fn mismatched_lengths_are_rejected() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("invariant: kernels compile");

    let short = device.upload(&ascending(8)).expect("invariant: upload");
    let long = device.upload(&ascending(16)).expect("invariant: upload");

    assert!(matches!(
        ops.axpy(&device, &short, &long, 1.0),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        ops.xpay(&device, &short, &long, 1.0),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        ops.copy_vector(&device, &short, &long),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        ops.subtract_into(&device, &short, &long, &short),
        Err(HephaestusError::LengthMismatch { .. })
    ));
}
