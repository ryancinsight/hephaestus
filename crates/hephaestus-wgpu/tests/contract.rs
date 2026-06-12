//! Differential contract tests: wgpu dispatch vs CPU reference.
//!
//! Tests acquire a real adapter; on hosts without one (headless CI without
//! GPU/lavapipe) they skip with a message rather than fabricate a pass.

use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{
    binary_elementwise, binary_elementwise_into, reduction, scalar_elementwise,
    scalar_elementwise_into, unary_elementwise, unary_elementwise_into, AbsOp, AddOp,
    ComputeDevice, DeviceBuffer, ExpOp, HephaestusError, MaxOp, MinOp, MulOp, NegOp, RecipOp,
    SqrtOp, SubOp, SumOp, WgpuDevice,
};

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-contract-test") {
        Ok(device) => Some(device),
        Err(e) => {
            eprintln!("skipping wgpu contract test: {e}");
            None
        }
    }
}

fn assert_elementwise_alias_rejected(result: hephaestus_wgpu::Result<()>) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => {
            assert!(
                message.starts_with("output buffer must not alias "),
                "unexpected alias rejection message: {message}"
            );
        }
        other => panic!("expected elementwise alias rejection, got {other:?}"),
    }
}

#[test]
fn upload_download_round_trips_values() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host: Vec<f32> = (0..1027).map(|i| i as f32 * 0.5 - 100.0).collect();
    let buffer = device.upload(&host).unwrap();
    assert_eq!(buffer.len(), host.len());

    let mut out = vec![0.0f32; host.len()];
    device.download(&buffer, &mut out).unwrap();
    assert_eq!(out, host);
}

#[test]
fn download_rejects_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let buffer = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let mut out = vec![0.0f32; 2];
    assert!(matches!(
        device.download(&buffer, &mut out),
        Err(HephaestusError::LengthMismatch { .. })
    ));
}

#[test]
fn elementwise_add_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // 1027 elements: exercises a partial trailing workgroup (1027 = 4*256 + 3).
    let a_host: Vec<f32> = (0..1027).map(|i| i as f32 * 1.25).collect();
    let b_host: Vec<f32> = (0..1027).map(|i| 1000.0 - i as f32).collect();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = binary_elementwise::<AddOp, f32>(&device, &a, &b).unwrap();

    let mut got = vec![0.0f32; a_host.len()];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn elementwise_mul_matches_cpu_reference_integral() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let a_host: Vec<u32> = (0..513).collect();
    let b_host: Vec<u32> = (0..513).map(|i| i + 7).collect();
    let expected: Vec<u32> = a_host.iter().zip(&b_host).map(|(x, y)| x * y).collect();

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = binary_elementwise::<MulOp, u32>(&device, &a, &b).unwrap();

    let mut got = vec![0u32; a_host.len()];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn elementwise_rejects_input_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let a = device.upload(&[1.0f32, 2.0]).unwrap();
    let b = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert!(binary_elementwise::<AddOp, f32>(&device, &a, &b).is_err());
}

#[test]
fn elementwise_into_reuses_caller_output_buffers() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a_host: Vec<f32> = (0..513).map(|i| i as f32 * 0.25).collect();
    let b_host: Vec<f32> = (0..513).map(|i| 50.0 - i as f32).collect();
    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(a_host.len()).unwrap();

    binary_elementwise_into::<SubOp, f32>(&device, &a, &b, &out, width).unwrap();
    let mut got = vec![0.0f32; a_host.len()];
    device.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x - y).collect();
    assert_eq!(got, expected);

    unary_elementwise_into::<NegOp, f32>(&device, &a, &out, width).unwrap();
    device.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| -x).collect();
    assert_eq!(got, expected);

    scalar_elementwise_into::<AddOp, f32>(&device, &a, 7.5, &out, width).unwrap();
    device.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| x + 7.5).collect();
    assert_eq!(got, expected);

    let short = device.alloc_zeroed::<f32>(a_host.len() - 1).unwrap();
    assert!(matches!(
        binary_elementwise_into::<AddOp, f32>(&device, &a, &b, &short, width),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        unary_elementwise_into::<NegOp, f32>(&device, &a, &short, width),
        Err(HephaestusError::LengthMismatch { .. })
    ));
    assert!(matches!(
        scalar_elementwise_into::<AddOp, f32>(&device, &a, 1.0, &short, width),
        Err(HephaestusError::LengthMismatch { .. })
    ));
}

#[test]
fn elementwise_into_rejects_output_input_aliasing() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let b = device.upload(&[4.0f32, 5.0, 6.0]).unwrap();

    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &device, &a, &b, &a, width,
    ));
    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &device, &a, &b, &b, width,
    ));
    assert_elementwise_alias_rejected(unary_elementwise_into::<NegOp, f32>(&device, &a, &a, width));
    assert_elementwise_alias_rejected(scalar_elementwise_into::<AddOp, f32>(
        &device, &a, 1.0, &a, width,
    ));
}

#[test]
fn elementwise_unary_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = vec![-4.0f32, -1.0, 0.0, 2.0, 16.0];
    let a = device.upload(&host).unwrap();

    // SqrtOp (note: sqrt(-4.0) and sqrt(-1.0) on f32 produce NaN, we compare matching values manually)
    let out_sqrt = unary_elementwise::<SqrtOp, f32>(&device, &a).unwrap();
    let mut got_sqrt = vec![0.0f32; host.len()];
    device.download(&out_sqrt, &mut got_sqrt).unwrap();
    assert!(got_sqrt[0].is_nan());
    assert!(got_sqrt[1].is_nan());
    assert_eq!(got_sqrt[2], 0.0f32);
    assert_eq!(got_sqrt[3], std::f32::consts::SQRT_2);
    assert_eq!(got_sqrt[4], 4.0f32);

    // AbsOp
    let out_abs = unary_elementwise::<AbsOp, f32>(&device, &a).unwrap();
    let mut got_abs = vec![0.0f32; host.len()];
    device.download(&out_abs, &mut got_abs).unwrap();
    assert_eq!(got_abs, vec![4.0f32, 1.0, 0.0, 2.0, 16.0]);

    // NegOp
    let out_neg = unary_elementwise::<NegOp, f32>(&device, &a).unwrap();
    let mut got_neg = vec![0.0f32; host.len()];
    device.download(&out_neg, &mut got_neg).unwrap();
    assert_eq!(got_neg, vec![4.0f32, 1.0, 0.0, -2.0, -16.0]);

    // ExpOp
    let out_exp = unary_elementwise::<ExpOp, f32>(&device, &a).unwrap();
    let mut got_exp = vec![0.0f32; host.len()];
    device.download(&out_exp, &mut got_exp).unwrap();
    for (i, &x) in host.iter().enumerate() {
        let expected = x.exp();
        let diff = (got_exp[i] - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            diff < tolerance,
            "Exp mismatch at index {}: got {}, expected {}, diff {}, tol {}",
            i,
            got_exp[i],
            expected,
            diff,
            tolerance
        );
    }

    // RecipOp
    let host_recip = vec![1.0f32, 2.0, 4.0, 8.0];
    let b = device.upload(&host_recip).unwrap();
    let out_recip = unary_elementwise::<RecipOp, f32>(&device, &b).unwrap();
    let mut got_recip = vec![0.0f32; host_recip.len()];
    device.download(&out_recip, &mut got_recip).unwrap();
    assert_eq!(got_recip, vec![1.0f32, 0.5, 0.25, 0.125]);
}

#[test]
fn elementwise_scalar_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let a = device.upload(&host).unwrap();

    // scalar add
    let out_add = scalar_elementwise::<AddOp, f32>(&device, &a, 10.0).unwrap();
    let mut got_add = vec![0.0f32; host.len()];
    device.download(&out_add, &mut got_add).unwrap();
    assert_eq!(got_add, vec![11.0f32, 12.0, 13.0, 14.0, 15.0]);

    // scalar mul
    let out_mul = scalar_elementwise::<MulOp, f32>(&device, &a, 3.0).unwrap();
    let mut got_mul = vec![0.0f32; host.len()];
    device.download(&out_mul, &mut got_mul).unwrap();
    assert_eq!(got_mul, vec![3.0f32, 6.0, 9.0, 12.0, 15.0]);
}

#[test]
fn reduction_sum_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
        // f32
        let host_f32: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let expected_f32: f32 = host_f32.iter().sum();
        let buf_f32 = device.upload(&host_f32).unwrap();
        let out_f32 = reduction::<SumOp, f32>(&device, &buf_f32).unwrap();
        let mut got_f32 = vec![0.0f32; 1];
        device.download(&out_f32, &mut got_f32).unwrap();
        assert_eq!(
            got_f32[0], expected_f32,
            "f32 sum mismatch at size {}",
            size
        );

        // u32
        let host_u32: Vec<u32> = (0..size).map(|i| i as u32).collect();
        let expected_u32: u32 = host_u32.iter().sum();
        let buf_u32 = device.upload(&host_u32).unwrap();
        let out_u32 = reduction::<SumOp, u32>(&device, &buf_u32).unwrap();
        let mut got_u32 = vec![0u32; 1];
        device.download(&out_u32, &mut got_u32).unwrap();
        assert_eq!(
            got_u32[0], expected_u32,
            "u32 sum mismatch at size {}",
            size
        );

        // i32
        let host_i32: Vec<i32> = (0..size).map(|i| if i % 2 == 0 { i } else { -i }).collect();
        let expected_i32: i32 = host_i32.iter().sum();
        let buf_i32 = device.upload(&host_i32).unwrap();
        let out_i32 = reduction::<SumOp, i32>(&device, &buf_i32).unwrap();
        let mut got_i32 = vec![0i32; 1];
        device.download(&out_i32, &mut got_i32).unwrap();
        assert_eq!(
            got_i32[0], expected_i32,
            "i32 sum mismatch at size {}",
            size
        );
    }
}

#[test]
fn reduction_min_max_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
        // f32 Min/Max
        let host_f32: Vec<f32> = (0..size)
            .map(|i| (i as f32 * 12.34 - 100.0).sin())
            .collect();
        let expected_min_f32 = if size == 0 {
            f32::MAX
        } else {
            host_f32.iter().copied().fold(f32::NAN, f32::min)
        };
        let expected_max_f32 = if size == 0 {
            f32::MIN
        } else {
            host_f32.iter().copied().fold(f32::NAN, f32::max)
        };

        let buf_f32 = device.upload(&host_f32).unwrap();

        let out_min_f32 = reduction::<MinOp, f32>(&device, &buf_f32).unwrap();
        let mut got_min_f32 = vec![0.0f32; 1];
        device.download(&out_min_f32, &mut got_min_f32).unwrap();
        assert_eq!(
            got_min_f32[0], expected_min_f32,
            "f32 min mismatch at size {}",
            size
        );

        let out_max_f32 = reduction::<MaxOp, f32>(&device, &buf_f32).unwrap();
        let mut got_max_f32 = vec![0.0f32; 1];
        device.download(&out_max_f32, &mut got_max_f32).unwrap();
        assert_eq!(
            got_max_f32[0], expected_max_f32,
            "f32 max mismatch at size {}",
            size
        );

        // i32 Min/Max
        let host_i32: Vec<i32> = (0..size)
            .map(|i| if i % 3 == 0 { i * 7 } else { -(i * 5) })
            .collect();
        let expected_min_i32 = if size == 0 {
            i32::MAX
        } else {
            *host_i32.iter().min().unwrap()
        };
        let expected_max_i32 = if size == 0 {
            i32::MIN
        } else {
            *host_i32.iter().max().unwrap()
        };

        let buf_i32 = device.upload(&host_i32).unwrap();

        let out_min_i32 = reduction::<MinOp, i32>(&device, &buf_i32).unwrap();
        let mut got_min_i32 = vec![0i32; 1];
        device.download(&out_min_i32, &mut got_min_i32).unwrap();
        assert_eq!(
            got_min_i32[0], expected_min_i32,
            "i32 min mismatch at size {}",
            size
        );

        let out_max_i32 = reduction::<MaxOp, i32>(&device, &buf_i32).unwrap();
        let mut got_max_i32 = vec![0i32; 1];
        device.download(&out_max_i32, &mut got_max_i32).unwrap();
        assert_eq!(
            got_max_i32[0], expected_max_i32,
            "i32 max mismatch at size {}",
            size
        );
    }
}

#[test]
fn acquisition_reports_themis_topology_from_adapter() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let topology = device
        .topology()
        .expect("acquisition path must capture a topology snapshot");

    // Differential against the API itself: re-query the same default
    // high-performance adapter and compare the reported fields.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter acquired above");
    assert_eq!(topology.warp_width(), adapter.limits().min_subgroup_size);
    let expected_tier = match adapter.get_info().device_type {
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => themis::MemoryTier::Dram,
        _ => themis::MemoryTier::Device,
    };
    assert_eq!(topology.memory_tier(), expected_tier);

    // Unreported-by-wgpu capacities must be zero, never fabricated.
    assert_eq!(topology.compute_units(), 0);
    assert_eq!(topology.registers_per_unit(), 0);
    assert_eq!(topology.shared_mem_per_unit_bytes(), 0);

    // The Arc-wrapping constructor has no adapter and reports none.
    let wrapped = WgpuDevice::new(device.device().clone(), device.queue().clone());
    assert!(wrapped.topology().is_none());
}
