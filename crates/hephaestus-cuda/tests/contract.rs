//! Contract tests for the CUDA `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without the `cuda` feature or without a CUDA device,
//! [`CudaDevice::try_default`] returns `Err` and each test skips.

use hephaestus_core::{
    BlockWidth, ComputeDevice, ComputeDeviceCapabilities, DeviceBuffer, DeviceFeature,
    HephaestusError, Result,
};
use hephaestus_cuda::{
    binary_elementwise, binary_elementwise_into, det, dot, kron, matexp, matmul, matmul_into,
    matrix_rank, matrix_rank_with_tolerance, norm_l1, norm_l2, norm_max, pinv, reduce_axis,
    reduction, reduction_with_width, scalar_elementwise, scalar_elementwise_into, scan_axis, trace,
    unary_elementwise, unary_elementwise_into, AbsOp, AddOp, CudaDevice, CumSumOp, ExpOp, MaxOp,
    MinOp, MulOp, NegOp, RecipOp, SqrtOp, StridedOperand, SubOp, SumOp,
};
use leto::Layout;

/// Acquire a device, or `None` to skip (no `cuda` feature / no GPU).
fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skip {test}: CUDA device unavailable ({e})");
            None
        }
    }
}

fn assert_elementwise_alias_rejected(result: Result<()>) {
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

fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
    match result {
        Err(HephaestusError::LengthMismatch {
            host_len: got_host,
            device_len: got_device,
        }) => {
            assert_eq!(got_host, host_len);
            assert_eq!(got_device, device_len);
        }
        Err(error) => panic!("expected length mismatch {host_len}->{device_len}, got {error:?}"),
        Ok(_) => panic!("expected length mismatch {host_len}->{device_len}, got success"),
    }
}

fn assert_dispatch_message<T>(result: Result<T>, expected: &str) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => assert_eq!(message, expected),
        Err(error) => panic!("expected dispatch failure {expected:?}, got {error:?}"),
        Ok(_) => panic!("expected dispatch failure {expected:?}, got success"),
    }
}

#[test]
fn device_capabilities_are_driver_backed() {
    let Some(dev) = device("device_capabilities_are_driver_backed") else {
        return;
    };

    let limits = dev.device_limits();
    assert!(limits.max_buffer_size > 0);
    assert!(limits.max_compute_workgroup_size_x > 0);
    assert!(limits.max_compute_workgroup_size_y > 0);
    assert!(limits.max_compute_workgroup_size_z > 0);
    assert!(limits.max_compute_invocations_per_workgroup > 0);
    assert!(limits.max_compute_workgroup_storage_size > 0);
    assert_eq!(limits.max_storage_buffers_per_shader_stage, None);
    assert_eq!(limits.max_push_constant_size, 0);

    assert!(dev.supports_device_feature(DeviceFeature::PushConstants));
    assert!(!dev.supports_device_feature(DeviceFeature::TimestampQuery));
    assert!(!dev.supports_device_feature(DeviceFeature::ShaderF16));
    assert!(!dev.supports_device_feature(DeviceFeature::MappablePrimaryBuffers));
}

#[test]
fn upload_download_roundtrip_f32() {
    let Some(dev) = device("upload_download_roundtrip_f32") else {
        return;
    };
    assert_eq!(dev.backend_name(), "cuda");
    let host = vec![1.0f32, 2.0, -3.5, 4.25, 0.0, 1024.5];
    let buf = dev.upload(&host).expect("upload");
    assert_eq!(buf.len(), host.len());
    let mut out = vec![0.0f32; host.len()];
    dev.download(&buf, &mut out).expect("download");
    assert_eq!(out, host, "round-trip must be the identity");
}

#[test]
fn test_placement_aware_allocation() {
    let Some(dev) = device("test_placement_aware_allocation") else {
        return;
    };
    use themis::{MemoryTier, PlacementHint};

    // CUDA primary buffers use non-managed `cuMemAlloc_v2` device memory even
    // when a host-visible placement hint is supplied.
    let hint = PlacementHint::Tier(MemoryTier::HostPinned);
    let buf1 = dev.alloc_zeroed_with_hint::<f32>(128, hint).unwrap();
    assert_eq!(buf1.len(), 128);
    assert_eq!(buf1.tier(), MemoryTier::Device);

    let host = vec![1.5f32; 128];
    let buf2 = dev.upload_with_hint(&host, hint).unwrap();
    assert_eq!(buf2.len(), 128);
    assert_eq!(buf2.tier(), MemoryTier::Device);

    // Test Dram / unified host memory hints normalize to the implemented
    // non-managed device tier.
    let hint_dram = PlacementHint::Tier(MemoryTier::Dram);
    let buf3 = dev.alloc_zeroed_with_hint::<f32>(128, hint_dram).unwrap();
    assert_eq!(buf3.tier(), MemoryTier::Device);

    let registers =
        dev.alloc_zeroed_with_hint::<f32>(128, PlacementHint::Tier(MemoryTier::Registers));
    match registers {
        Err(HephaestusError::AllocationFailed { message }) => assert_eq!(
            message,
            "CUDA primary buffers cannot be allocated from budget-only tier Registers"
        ),
        other => panic!("expected budget-only tier rejection, got {other:?}"),
    }

    // Test default non-hinted delegates
    let buf4 = dev.alloc_zeroed::<f32>(128).unwrap();
    assert_eq!(buf4.tier(), MemoryTier::Device);
}

#[test]
fn upload_download_roundtrip_i32() {
    let Some(dev) = device("upload_download_roundtrip_i32") else {
        return;
    };
    let host: Vec<i32> = (-4..=4).collect();
    let buf = dev.upload(&host).expect("upload");
    let mut out = vec![0i32; host.len()];
    dev.download(&buf, &mut out).expect("download");
    assert_eq!(out, host);
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
fn empty_buffer_roundtrips() {
    let Some(dev) = device("empty_buffer_roundtrips") else {
        return;
    };
    let buf = dev.upload::<f32>(&[]).expect("upload empty");
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
    let mut out: Vec<f32> = Vec::new();
    dev.download(&buf, &mut out).expect("download empty");
}

#[test]
fn download_length_mismatch_rejected() {
    let Some(dev) = device("download_length_mismatch_rejected") else {
        return;
    };
    let buf = dev.upload(&[1.0f32, 2.0]).expect("upload");
    let mut out = vec![0.0f32; 5];
    let err = dev
        .download(&buf, &mut out)
        .expect_err("length mismatch must be rejected");
    assert!(
        matches!(
            err,
            HephaestusError::LengthMismatch {
                host_len: 5,
                device_len: 2,
            }
        ),
        "expected LengthMismatch{{5, 2}}, got {err:?}"
    );
}

#[test]
fn elementwise_add_matches_cpu_reference() {
    let Some(dev) = device("elementwise_add_matches_cpu_reference") else {
        return;
    };
    let a_host: Vec<f32> = (0..1027).map(|i| i as f32 * 1.25).collect();
    let b_host: Vec<f32> = (0..1027).map(|i| 1000.0 - i as f32).collect();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = binary_elementwise::<AddOp, f32>(&dev, &a, &b).unwrap();

    let mut got = vec![0.0f32; a_host.len()];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn elementwise_mul_matches_cpu_reference_integral() {
    let Some(dev) = device("elementwise_mul_matches_cpu_reference_integral") else {
        return;
    };
    let a_host: Vec<u32> = (0..513).collect();
    let b_host: Vec<u32> = (0..513).map(|i| i + 7).collect();
    let expected: Vec<u32> = a_host.iter().zip(&b_host).map(|(x, y)| x * y).collect();

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = binary_elementwise::<MulOp, u32>(&dev, &a, &b).unwrap();

    let mut got = vec![0u32; a_host.len()];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn elementwise_rejects_input_length_mismatch() {
    let Some(dev) = device("elementwise_rejects_input_length_mismatch") else {
        return;
    };
    let a = dev.upload(&[1.0f32, 2.0]).unwrap();
    let b = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(binary_elementwise::<AddOp, f32>(&dev, &a, &b), 2, 3);
}

#[test]
fn elementwise_into_reuses_caller_output_buffers() {
    let Some(dev) = device("elementwise_into_reuses_caller_output_buffers") else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a_host: Vec<f32> = (0..513).map(|i| i as f32 * 0.25).collect();
    let b_host: Vec<f32> = (0..513).map(|i| 50.0 - i as f32).collect();
    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(a_host.len()).unwrap();

    binary_elementwise_into::<SubOp, f32>(&dev, &a, &b, &out, width).unwrap();
    let mut got = vec![0.0f32; a_host.len()];
    dev.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x - y).collect();
    assert_eq!(got, expected);

    unary_elementwise_into::<NegOp, f32>(&dev, &a, &out, width).unwrap();
    dev.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| -x).collect();
    assert_eq!(got, expected);

    scalar_elementwise_into::<AddOp, f32>(&dev, &a, 7.5, &out, width).unwrap();
    dev.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| x + 7.5).collect();
    assert_eq!(got, expected);

    let short = dev.alloc_zeroed::<f32>(a_host.len() - 1).unwrap();
    assert_length_mismatch(
        binary_elementwise_into::<AddOp, f32>(&dev, &a, &b, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        unary_elementwise_into::<NegOp, f32>(&dev, &a, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        scalar_elementwise_into::<AddOp, f32>(&dev, &a, 1.0, &short, width),
        short.len(),
        a.len(),
    );
}

#[test]
fn elementwise_into_rejects_output_input_aliasing() {
    let Some(dev) = device("elementwise_into_rejects_output_input_aliasing") else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let b = dev.upload(&[4.0f32, 5.0, 6.0]).unwrap();

    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &dev, &a, &b, &a, width,
    ));
    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &dev, &a, &b, &b, width,
    ));
    assert_elementwise_alias_rejected(unary_elementwise_into::<NegOp, f32>(&dev, &a, &a, width));
    assert_elementwise_alias_rejected(scalar_elementwise_into::<AddOp, f32>(
        &dev, &a, 1.0, &a, width,
    ));
}

#[test]
fn elementwise_unary_matches_cpu_reference() {
    let Some(dev) = device("elementwise_unary_matches_cpu_reference") else {
        return;
    };
    let host = vec![-4.0f32, -1.0, 0.0, 2.0, 16.0];
    let a = dev.upload(&host).unwrap();

    // SqrtOp
    let out_sqrt = unary_elementwise::<SqrtOp, f32>(&dev, &a).unwrap();
    let mut got_sqrt = vec![0.0f32; host.len()];
    dev.download(&out_sqrt, &mut got_sqrt).unwrap();
    assert!(got_sqrt[0].is_nan());
    assert!(got_sqrt[1].is_nan());
    assert_eq!(got_sqrt[2], 0.0f32);
    assert_eq!(got_sqrt[3], std::f32::consts::SQRT_2);
    assert_eq!(got_sqrt[4], 4.0f32);

    // AbsOp
    let out_abs = unary_elementwise::<AbsOp, f32>(&dev, &a).unwrap();
    let mut got_abs = vec![0.0f32; host.len()];
    dev.download(&out_abs, &mut got_abs).unwrap();
    assert_eq!(got_abs, vec![4.0f32, 1.0, 0.0, 2.0, 16.0]);

    // NegOp
    let out_neg = unary_elementwise::<NegOp, f32>(&dev, &a).unwrap();
    let mut got_neg = vec![0.0f32; host.len()];
    dev.download(&out_neg, &mut got_neg).unwrap();
    assert_eq!(got_neg, vec![4.0f32, 1.0, 0.0, -2.0, -16.0]);

    // ExpOp
    let out_exp = unary_elementwise::<ExpOp, f32>(&dev, &a).unwrap();
    let mut got_exp = vec![0.0f32; host.len()];
    dev.download(&out_exp, &mut got_exp).unwrap();
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
    let b = dev.upload(&host_recip).unwrap();
    let out_recip = unary_elementwise::<RecipOp, f32>(&dev, &b).unwrap();
    let mut got_recip = vec![0.0f32; host_recip.len()];
    dev.download(&out_recip, &mut got_recip).unwrap();
    assert_eq!(got_recip, vec![1.0f32, 0.5, 0.25, 0.125]);
}

#[test]
fn elementwise_scalar_matches_cpu_reference() {
    let Some(dev) = device("elementwise_scalar_matches_cpu_reference") else {
        return;
    };
    let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let a = dev.upload(&host).unwrap();

    let out_add = scalar_elementwise::<AddOp, f32>(&dev, &a, 10.0).unwrap();
    let mut got_add = vec![0.0f32; host.len()];
    dev.download(&out_add, &mut got_add).unwrap();
    assert_eq!(got_add, vec![11.0f32, 12.0, 13.0, 14.0, 15.0]);

    let out_mul = scalar_elementwise::<MulOp, f32>(&dev, &a, 3.0).unwrap();
    let mut got_mul = vec![0.0f32; host.len()];
    dev.download(&out_mul, &mut got_mul).unwrap();
    assert_eq!(got_mul, vec![3.0f32, 6.0, 9.0, 12.0, 15.0]);
}

#[test]
fn reduction_sum_matches_cpu_reference() {
    let Some(dev) = device("reduction_sum_matches_cpu_reference") else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
        // f32
        let host_f32: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let expected_f32: f32 = host_f32.iter().sum();
        let buf_f32 = dev.upload(&host_f32).unwrap();
        let out_f32 = reduction::<SumOp, f32>(&dev, &buf_f32).unwrap();
        let mut got_f32 = vec![0.0f32; 1];
        dev.download(&out_f32, &mut got_f32).unwrap();
        assert_eq!(
            got_f32[0], expected_f32,
            "f32 sum mismatch at size {}",
            size
        );

        // u32
        let host_u32: Vec<u32> = (0..size).map(|i| i as u32).collect();
        let expected_u32: u32 = host_u32.iter().sum();
        let buf_u32 = dev.upload(&host_u32).unwrap();
        let out_u32 = reduction::<SumOp, u32>(&dev, &buf_u32).unwrap();
        let mut got_u32 = vec![0u32; 1];
        dev.download(&out_u32, &mut got_u32).unwrap();
        assert_eq!(
            got_u32[0], expected_u32,
            "u32 sum mismatch at size {}",
            size
        );

        // i32
        let host_i32: Vec<i32> = (0..size).map(|i| if i % 2 == 0 { i } else { -i }).collect();
        let expected_i32: i32 = host_i32.iter().sum();
        let buf_i32 = dev.upload(&host_i32).unwrap();
        let out_i32 = reduction::<SumOp, i32>(&dev, &buf_i32).unwrap();
        let mut got_i32 = vec![0i32; 1];
        dev.download(&out_i32, &mut got_i32).unwrap();
        assert_eq!(
            got_i32[0], expected_i32,
            "i32 sum mismatch at size {}",
            size
        );
    }
}

#[test]
fn reduction_min_max_matches_cpu_reference() {
    let Some(dev) = device("reduction_min_max_matches_cpu_reference") else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
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

        let buf_f32 = dev.upload(&host_f32).unwrap();

        let out_min_f32 = reduction::<MinOp, f32>(&dev, &buf_f32).unwrap();
        let mut got_min_f32 = vec![0.0f32; 1];
        dev.download(&out_min_f32, &mut got_min_f32).unwrap();
        assert_eq!(
            got_min_f32[0], expected_min_f32,
            "f32 min mismatch at size {}",
            size
        );

        let out_max_f32 = reduction::<MaxOp, f32>(&dev, &buf_f32).unwrap();
        let mut got_max_f32 = vec![0.0f32; 1];
        dev.download(&out_max_f32, &mut got_max_f32).unwrap();
        assert_eq!(
            got_max_f32[0], expected_max_f32,
            "f32 max mismatch at size {}",
            size
        );
    }
}

#[test]
fn reduction_width_is_part_of_dispatch_contract() {
    let Some(dev) = device("reduction_width_is_part_of_dispatch_contract") else {
        return;
    };

    let host: Vec<u32> = (0..1027).collect();
    let expected: u32 = host.iter().sum();
    let input = dev.upload(&host).unwrap();

    let narrow = BlockWidth::new(128).unwrap();
    let out_narrow = reduction_with_width::<SumOp, u32>(&dev, &input, narrow).unwrap();
    let mut got_narrow = vec![0u32; 1];
    dev.download(&out_narrow, &mut got_narrow).unwrap();
    assert_eq!(got_narrow[0], expected);

    let non_power = BlockWidth::new(192).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&dev, &input, non_power),
        "reduction block width 192 must be a power of two",
    );
}

#[test]
fn linalg_matmul_matches_cpu_reference() {
    let Some(dev) = device("linalg_matmul_matches_cpu_reference") else {
        return;
    };

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_host = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let expected = vec![
        29.0f32, 32.0, 35.0, 38.0, 65.0, 72.0, 79.0, 86.0, 101.0, 112.0, 123.0, 134.0,
    ];

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(12).unwrap();

    let a_layout = Layout::c_contiguous([3, 2]).unwrap();
    let b_layout = Layout::c_contiguous([2, 4]).unwrap();
    let out_layout = Layout::c_contiguous([3, 4]).unwrap();

    matmul_into(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        StridedOperand {
            buffer: &b,
            layout: &b_layout,
        },
        StridedOperand {
            buffer: &out,
            layout: &out_layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 12];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn linalg_dot_matches_cpu_reference() {
    let Some(dev) = device("linalg_dot_matches_cpu_reference") else {
        return;
    };

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
    let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();

    let a_layout = Layout::c_contiguous([4]).unwrap();
    let b_layout = Layout::c_contiguous([4]).unwrap();

    let out_buf = dot(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        StridedOperand {
            buffer: &b,
            layout: &b_layout,
        },
    )
    .unwrap();

    let mut got = [0.0f32; 1];
    dev.download(&out_buf, &mut got).unwrap();
    assert_eq!(got[0], expected);
}

#[test]
fn linalg_trace_matches_cpu_reference() {
    let Some(dev) = device("linalg_trace_matches_cpu_reference") else {
        return;
    };

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected = 1.0 + 5.0 + 9.0;

    let a = dev.upload(&a_host).unwrap();
    let a_layout = Layout::c_contiguous([3, 3]).unwrap();

    let out_buf = trace(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
    )
    .unwrap();

    let mut got = [0.0f32; 1];
    dev.download(&out_buf, &mut got).unwrap();
    assert_eq!(got[0], expected);
}

#[test]
fn linalg_norms_match_cpu_reference() {
    let Some(dev) = device("linalg_norms_match_cpu_reference") else {
        return;
    };

    let a_host = vec![-1.0f32, 2.0, -3.0, 4.0];
    let a = dev.upload(&a_host).unwrap();
    let a_layout = Layout::c_contiguous([4]).unwrap();
    let operand = StridedOperand {
        buffer: &a,
        layout: &a_layout,
    };

    // L1
    let l1_buf = norm_l1(&dev, operand).unwrap();
    let mut got_l1 = [0.0f32; 1];
    dev.download(&l1_buf, &mut got_l1).unwrap();
    assert_eq!(got_l1[0], 10.0);

    // L2
    let l2_buf = norm_l2(&dev, operand).unwrap();
    let mut got_l2 = [0.0f32; 1];
    dev.download(&l2_buf, &mut got_l2).unwrap();
    let expected_l2 = 30.0f32.sqrt();
    assert!((got_l2[0] - expected_l2).abs() <= 1e-5);

    // Max
    let max_buf = norm_max(&dev, operand).unwrap();
    let mut got_max = [0.0f32; 1];
    dev.download(&max_buf, &mut got_max).unwrap();
    assert_eq!(got_max[0], 4.0);
}

#[test]
fn linalg_matmul_allocating_matches_cpu_reference() {
    let Some(dev) = device("linalg_matmul_allocating_matches_cpu_reference") else {
        return;
    };

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_host = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let expected = vec![
        29.0f32, 32.0, 35.0, 38.0, 65.0, 72.0, 79.0, 86.0, 101.0, 112.0, 123.0, 134.0,
    ];

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();

    let a_layout = Layout::c_contiguous([3, 2]).unwrap();
    let b_layout = Layout::c_contiguous([2, 4]).unwrap();

    let out = matmul(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        StridedOperand {
            buffer: &b,
            layout: &b_layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 12];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn linalg_kron_matches_cpu_reference() {
    let Some(dev) = device("linalg_kron_matches_cpu_reference") else {
        return;
    };

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
    let expected = vec![
        5.0f32, 6.0, 10.0, 12.0, 7.0, 8.0, 14.0, 16.0, 15.0, 18.0, 20.0, 24.0, 21.0, 24.0, 28.0,
        32.0,
    ];

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();

    let a_layout = Layout::c_contiguous([2, 2]).unwrap();
    let b_layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = kron(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        StridedOperand {
            buffer: &b,
            layout: &b_layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 16];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn linalg_pinv_matches_closed_form_diagonal() {
    let Some(dev) = device("linalg_pinv_matches_closed_form_diagonal") else {
        return;
    };

    let matrix_host = vec![2.0f32, 0.0, 0.0, 4.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = pinv(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 4];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![0.5, 0.0, 0.0, 0.25]);
}

#[test]
fn linalg_matexp_matches_closed_form_diagonal() {
    let Some(dev) = device("linalg_matexp_matches_closed_form_diagonal") else {
        return;
    };

    let matrix_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = matexp(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 4];
    dev.download(&out, &mut got).unwrap();
    let expected = [1.0f32, 0.0, 0.0, 1.0f32.exp()];
    for (index, (&actual, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        let tolerance = 64.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "matrix exponential mismatch at {index}: got {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

#[test]
fn reduction_axis_reduction_generic_matches_cpu() {
    let Some(dev) = device("reduction_axis_reduction_generic_matches_cpu") else {
        return;
    };

    let host_in = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = dev.upload(&host_in).unwrap();
    let a_layout = Layout::c_contiguous([2, 3]).unwrap();

    // Sum axis 0
    let out = reduce_axis::<SumOp, f32>(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = vec![0.0f32; 3];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![5.0, 7.0, 9.0]);
}

#[test]
fn scan_scan_axis_matches_cpu() {
    let Some(dev) = device("scan_scan_axis_matches_cpu") else {
        return;
    };

    let host_in = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = dev.upload(&host_in).unwrap();
    let a_layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = scan_axis::<CumSumOp, f32>(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        1,
        hephaestus_cuda::ScanDirection::Forward,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = vec![0.0f32; 4];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![1.0, 3.0, 3.0, 7.0]);
}

#[test]
fn linalg_matrix_rank_matches_reference() {
    let Some(dev) = device("linalg_matrix_rank_matches_reference") else {
        return;
    };

    // Diagonal matrix with rank 2
    let host_in = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    let a = dev.upload(&host_in).unwrap();
    let a_layout = Layout::c_contiguous([3, 3]).unwrap();

    let rank = matrix_rank(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
    )
    .unwrap();
    assert_eq!(rank, 2);

    let rank_tol = matrix_rank_with_tolerance(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        0.5,
    )
    .unwrap();
    assert_eq!(rank_tol, 1);
}

#[test]
fn linalg_det_matches_reference() {
    let Some(dev) = device("linalg_det_matches_reference") else {
        return;
    };

    // Diagonal matrix with determinant = 3.0 * 2.0 * -1.0 = -6.0
    let host_in = vec![3.0f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0];
    let a = dev.upload(&host_in).unwrap();
    let a_layout = Layout::c_contiguous([3, 3]).unwrap();

    let det_buffer = det(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
    )
    .unwrap();
    let mut got = [0.0f32; 1];
    dev.download(&det_buffer, &mut got).unwrap();
    assert!((got[0] - (-6.0f32)).abs() < 1.0e-5);
}

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_decomposition_matches_leto_reference() {
    let Some(dev) = device("cholesky_decomposition_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose;

    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![6.0f32, 5.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_cholesky = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_cholesky = cholesky_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_cholesky.n(), leto_cholesky.dim());
    assert_eq!(gpu_cholesky.det(), leto_cholesky.det());

    let mut got_lower = vec![0.0f32; 4];
    dev.download(gpu_cholesky.lower(), &mut got_lower).unwrap();
    let expected_lower = leto::Storage::as_slice(leto_cholesky.lower().storage());
    assert_eq!(got_lower, expected_lower);

    let solution = gpu_cholesky.solve(&dev, &rhs).unwrap();
    let expected_solution = leto_cholesky.solve(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; 2];
    dev.download(&solution, &mut got_solution).unwrap();
    assert_eq!(
        got_solution,
        leto::Storage::as_slice(expected_solution.storage())
    );

    let inverse = gpu_cholesky.inv(&dev).unwrap();
    let expected_inverse = leto_cholesky.inv().unwrap();
    let mut got_inverse = vec![0.0f32; 4];
    dev.download(&inverse, &mut got_inverse).unwrap();
    assert_eq!(
        got_inverse,
        leto::Storage::as_slice(expected_inverse.storage())
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_matches_leto_reference_across_block_boundary() {
    let Some(dev) = device("blocked_cholesky_matches_leto_reference_across_block_boundary") else {
        return;
    };
    use hephaestus_cuda::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    let n = 66usize;
    let mut matrix_host = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            matrix_host[row * n + col] = if row == col {
                n as f32 + 4.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host).unwrap();
    let leto_cholesky = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_cholesky = cholesky_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got_lower = vec![0.0f32; n * n];
    dev.download(gpu_cholesky.lower(), &mut got_lower).unwrap();
    let expected_lower = leto::Storage::as_slice(leto_cholesky.lower().storage());
    for (index, (&got, &expected)) in got_lower.iter().zip(expected_lower.iter()).enumerate() {
        let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "blocked Cholesky lower mismatch at {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
    assert_eq!(gpu_cholesky.det(), leto_cholesky.det());
}

#[cfg(feature = "decomposition")]
#[test]
fn lu_decomposition_matches_leto_reference() {
    let Some(dev) = device("lu_decomposition_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::lu_decompose;

    let matrix_host = vec![2.0f32, 1.0, 4.0, 3.0];
    let rhs_host = vec![3.0f32, 7.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.n(), leto_lu.dim());
    assert_eq!(gpu_lu.det(), leto_lu.det());

    let mut got_factors = vec![0.0f32; 4];
    dev.download(gpu_lu.factors(), &mut got_factors).unwrap();
    let expected_factors = leto::Storage::as_slice(leto_lu.factors().storage());
    assert_eq!(got_factors, expected_factors);

    let solution = gpu_lu.solve(&dev, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; 2];
    dev.download(&solution, &mut got_solution).unwrap();
    assert_eq!(
        got_solution,
        leto::Storage::as_slice(expected_solution.storage())
    );

    let inverse = gpu_lu.inv(&dev).unwrap();
    let expected_inverse = leto_lu.inv().unwrap();
    let mut got_inverse = vec![0.0f32; 4];
    dev.download(&inverse, &mut got_inverse).unwrap();
    assert_eq!(
        got_inverse,
        leto::Storage::as_slice(expected_inverse.storage())
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn qr_decomposition_matches_leto_reference() {
    let Some(dev) = device("qr_decomposition_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::qr_decompose;

    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([3], rhs_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), leto_qr.shape());

    let mut got_r = vec![0.0f32; 6];
    dev.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
    let expected_r = leto_qr.r();
    assert_eq!(got_r, leto::Storage::as_slice(expected_r.storage()));

    let solution = gpu_qr.solve_least_squares(&dev, &rhs).unwrap();
    let expected_solution = leto_qr.solve_least_squares(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; 2];
    dev.download(&solution, &mut got_solution).unwrap();
    assert_eq!(
        got_solution,
        leto::Storage::as_slice(expected_solution.storage())
    );

    let underdetermined = dev.alloc_zeroed::<f32>(6).unwrap();
    let underdetermined_layout = Layout::c_contiguous([2, 3]).unwrap();
    let underdetermined_qr = qr_decompose(
        &dev,
        StridedOperand {
            buffer: &underdetermined,
            layout: &underdetermined_layout,
        },
    );
    assert!(matches!(
        underdetermined_qr,
        Err(HephaestusError::DispatchFailed { message }) if message.contains("m ≥ n")
    ));
}

// ── write_buffer tests ────────────────────────────────────────────────

#[test]
fn write_buffer_overwrites_existing_data() {
    let Some(dev) = device("write_buffer_overwrites_existing_data") else {
        return;
    };

    // Upload initial data.
    let initial = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = dev.upload(&initial).unwrap();

    // Overwrite with new data via write_buffer.
    let updated = vec![10.0f32, 20.0, 30.0, 40.0];
    dev.write_buffer(&buf, &updated).unwrap();

    // Download and verify the overwritten data.
    let mut got = vec![0.0f32; 4];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, updated);
}

#[test]
fn write_buffer_rejects_length_mismatch() {
    let Some(dev) = device("write_buffer_rejects_length_mismatch") else {
        return;
    };

    let buf = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let wrong_len = vec![1.0f32, 2.0]; // len 2, buffer len 3
    assert_length_mismatch(dev.write_buffer(&buf, &wrong_len), 2, 3);
}

#[test]
fn write_buffer_empty_is_noop() {
    let Some(dev) = device("write_buffer_empty_is_noop") else {
        return;
    };

    let buf = dev.upload::<f32>(&[]).unwrap();
    dev.write_buffer(&buf, &[] as &[f32]).unwrap();
    assert_eq!(buf.len(), 0);
}

#[test]
fn write_buffer_integer_types() {
    let Some(dev) = device("write_buffer_integer_types") else {
        return;
    };

    let buf = dev.upload(&[0i32, 0, 0]).unwrap();
    let data = vec![42i32, -7, 100];
    dev.write_buffer(&buf, &data).unwrap();

    let mut got = vec![0i32; 3];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, data);
}

#[test]
fn write_sub_buffer_overwrites_only_requested_range() {
    let Some(dev) = device("write_sub_buffer_overwrites_only_requested_range") else {
        return;
    };

    let buf = dev.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    dev.write_sub_buffer(&buf, 1, &[20.0f32, 30.0]).unwrap();

    let mut got = [0.0f32; 4];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, [1.0, 20.0, 30.0, 4.0]);
}

#[test]
fn write_sub_buffer_rejects_out_of_range_write() {
    let Some(dev) = device("write_sub_buffer_rejects_out_of_range_write") else {
        return;
    };

    let buf = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(dev.write_sub_buffer(&buf, 2, &[4.0f32, 5.0]), 4, 3);
}

#[test]
fn write_sub_buffer_empty_tail_write_is_noop() {
    let Some(dev) = device("write_sub_buffer_empty_tail_write_is_noop") else {
        return;
    };

    let buf = dev.upload(&[9i32, 8, 7]).unwrap();
    dev.write_sub_buffer(&buf, 3, &[] as &[i32]).unwrap();

    let mut got = [0i32; 3];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, [9, 8, 7]);
}

// ── Extended differential decomposition tests ─────────────────────────────

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_identity_matrix_yields_identity_lower() {
    let Some(dev) = device("cholesky_identity_matrix_yields_identity_lower") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_chol = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_chol = cholesky_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_chol.n(), 2);
    assert_eq!(gpu_chol.det(), leto_chol.det());
    assert_eq!(gpu_chol.det(), 1.0);

    let mut got_lower = vec![0.0f32; 4];
    dev.download(gpu_chol.lower(), &mut got_lower).unwrap();
    let expected_lower = leto::Storage::as_slice(leto_chol.lower().storage());
    assert_eq!(got_lower, expected_lower);
    assert_eq!(got_lower, vec![1.0f32, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_spd_reconstruction_matches_original() {
    let Some(dev) = device("cholesky_spd_reconstruction_matches_original") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose;

    let matrix_host = vec![4.0f32, 2.0, 0.5, 2.0, 5.0, 1.0, 0.5, 1.0, 3.0];
    let n = 3;
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();

    let gpu_chol = cholesky_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got_lower = vec![0.0f32; n * n];
    dev.download(gpu_chol.lower(), &mut got_lower).unwrap();

    // Reconstruct A' = L * L^T and verify against original.
    for row in 0..n {
        for col in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                let l_rk = got_lower[row * n + k];
                let l_ck = got_lower[col * n + k]; // L^T[k, col] = L[col, k]
                sum += l_rk * l_ck;
            }
            let expected = matrix_host[row * n + col];
            let tolerance = 8.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (sum - expected).abs() <= tolerance,
                "Cholesky reconstruction mismatch at [{row},{col}]: got {sum}, expected {expected}"
            );
        }
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_solve_known_system_accurate() {
    let Some(dev) = device("cholesky_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose;

    // A = [[4, 2], [2, 3]], b = [8, 7]  =>  x = [1.75, 1.0]
    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![8.0f32, 7.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let gpu_chol = cholesky_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_chol.solve(&dev, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();
    assert!(
        (got[0] - 1.25f32).abs() <= 1e-5,
        "x[0] = {} expected 1.25",
        got[0]
    );
    assert!(
        (got[1] - 1.5f32).abs() <= 1e-5,
        "x[1] = {} expected 1.5",
        got[1]
    );

    let ax0 = 4.0 * got[0] + 2.0 * got[1];
    let ax1 = 2.0 * got[0] + 3.0 * got[1];
    assert!(
        (ax0 - 8.0).abs() <= 1e-4,
        "residual[0] = {} expected 8.0",
        ax0
    );
    assert!(
        (ax1 - 7.0).abs() <= 1e-4,
        "residual[1] = {} expected 7.0",
        ax1
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn cholesky_rejects_singular_matrix() {
    let Some(dev) = device("cholesky_rejects_singular_matrix") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(
        result.is_err(),
        "singular matrix must be rejected by Cholesky"
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn lu_identity_yields_identity_factors() {
    let Some(dev) = device("lu_identity_yields_identity_factors") else {
        return;
    };
    use hephaestus_cuda::lu_decompose;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.n(), 2);
    assert_eq!(gpu_lu.det(), leto_lu.det());
    assert_eq!(gpu_lu.det(), 1.0);

    let mut got_factors = vec![0.0f32; 4];
    dev.download(gpu_lu.factors(), &mut got_factors).unwrap();
    let expected_factors = leto::Storage::as_slice(leto_lu.factors().storage());
    assert_eq!(got_factors, expected_factors);
}

#[cfg(feature = "decomposition")]
#[test]
fn lu_solve_known_system_accurate() {
    let Some(dev) = device("lu_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::lu_decompose;

    // A = [[2, 1], [4, 3]], b = [5, 11]  =>  x = [2, 1]
    let matrix_host = vec![2.0f32, 1.0, 4.0, 3.0];
    let rhs_host = vec![5.0f32, 11.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_lu.solve(&dev, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();
    let expected = leto::Storage::as_slice(expected_solution.storage());
    for i in 0..2 {
        assert!(
            (got[i] - expected[i]).abs() <= 1e-5,
            "LU solve x[{i}] = {} expected {}",
            got[i],
            expected[i]
        );
    }
    assert!((got[0] - 2.0f32).abs() <= 1e-5);
    assert!((got[1] - 1.0f32).abs() <= 1e-5);
}

#[cfg(feature = "decomposition")]
#[test]
fn lu_rejects_singular_matrix() {
    let Some(dev) = device("lu_rejects_singular_matrix") else {
        return;
    };
    use hephaestus_cuda::lu_decompose;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = lu_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(result.is_err(), "singular matrix must be rejected by LU");
}

#[cfg(feature = "decomposition")]
#[test]
fn qr_identity_yields_identity_r() {
    let Some(dev) = device("qr_identity_yields_identity_r") else {
        return;
    };
    use hephaestus_cuda::qr_decompose;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (2, 2));
    assert_eq!(gpu_qr.shape(), leto_qr.shape());

    let mut got_r = vec![0.0f32; 4];
    dev.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
    let r_ref = leto_qr.r();
    let expected_r = leto::Storage::as_slice(r_ref.storage());
    assert_eq!(got_r, expected_r);
}

#[cfg(feature = "decomposition")]
#[test]
fn qr_solve_known_system_accurate() {
    let Some(dev) = device("qr_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::qr_decompose;

    // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]  =>  x = [1, 2]
    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();

    let gpu_qr = qr_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_qr.solve_least_squares(&dev, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();

    // Verify A*x ≈ b (residual check).
    let residual_0 = 1.0 * got[0] + 0.0 * got[1] - 1.0;
    let residual_1 = 0.0 * got[0] + 1.0 * got[1] - 2.0;
    let residual_2 = 1.0 * got[0] + 1.0 * got[1] - 3.0;
    assert!(residual_0.abs() <= 1e-4, "QR residual[0] = {residual_0}");
    assert!(residual_1.abs() <= 1e-4, "QR residual[1] = {residual_1}");
    assert!(residual_2.abs() <= 1e-4, "QR residual[2] = {residual_2}");
}

// ── Blocked decomposition differential tests ────────────────────────────

#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_matches_leto_reference() {
    let Some(dev) = device("blocked_lu_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{lu_decompose_blocked, StridedOperand};
    use leto::Layout;

    // 66×66 matrix exercises the block boundary (LU_BLOCK_SIZE = 64).
    let n = 66usize;
    let mut matrix_host = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            matrix_host[row * n + col] = if row == col {
                n as f32 + 4.0
            } else {
                0.1 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host.clone()).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    // The host-side inner decomposition (on original matrix) must match leto-ops.
    assert_eq!(gpu_lu.n(), leto_lu.dim());
    assert_eq!(gpu_lu.det(), leto_lu.det());

    // Solve via host-side decomposition must match.
    let rhs_host = vec![1.0f32; n];
    let rhs = dev.upload(&rhs_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([n], rhs_host).unwrap();
    let solution = gpu_lu.solve(&dev, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; n];
    dev.download(&solution, &mut got).unwrap();
    let expected = leto::Storage::as_slice(expected_solution.storage());
    for i in 0..n {
        assert!(
            (got[i] - expected[i]).abs() <= 1e-4,
            "blocked LU solve x[{i}] = {} expected {}",
            got[i],
            expected[i]
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_identity_yields_identity_factors() {
    let Some(dev) = device("blocked_lu_identity_yields_identity_factors") else {
        return;
    };
    use hephaestus_cuda::{lu_decompose_blocked, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.n(), 2);
    assert_eq!(gpu_lu.det(), leto_lu.det());
    assert_eq!(gpu_lu.det(), 1.0);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_solve_known_system_accurate() {
    let Some(dev) = device("blocked_lu_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::{lu_decompose_blocked, StridedOperand};
    use leto::Layout;

    // A = [[2, 1], [4, 3]], b = [5, 11]  =>  x = [2, 1]
    let matrix_host = vec![2.0f32, 1.0, 4.0, 3.0];
    let rhs_host = vec![5.0f32, 11.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_lu.solve(&dev, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();
    let expected = leto::Storage::as_slice(expected_solution.storage());
    for i in 0..2 {
        assert!(
            (got[i] - expected[i]).abs() <= 1e-5,
            "blocked LU solve x[{i}] = {} expected {}",
            got[i],
            expected[i]
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_rejects_singular_matrix() {
    let Some(dev) = device("blocked_lu_rejects_singular_matrix") else {
        return;
    };
    use hephaestus_cuda::{lu_decompose_blocked, StridedOperand};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = lu_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(
        result.is_err(),
        "singular matrix must be rejected by blocked LU"
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_matches_leto_reference() {
    let Some(dev) = device("blocked_qr_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{qr_decompose_blocked, StridedOperand};
    use leto::Layout;

    // 70×35 matrix exercises two QR blocks (QR_BLOCK_SIZE = 32).
    let (m, n) = (70, 35);
    let mut matrix_host = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            matrix_host[row * n + col] = if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([m, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([m, n], matrix_host.clone()).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (m, n));

    // R should be upper triangular — check lower triangle is zero.
    let mut got_r = vec![0.0f32; m * n];
    dev.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
    for i in 1..m {
        for j in 0..n.min(i) {
            assert!(
                got_r[i * n + j].abs() <= f32::EPSILON,
                "blocked QR R[{i},{j}] = {} should be zero (lower triangle)",
                got_r[i * n + j]
            );
        }
    }

    // Upper n×n block of R must match leto-ops.
    let leto_r = leto_qr.r();
    let expected_r = leto::Storage::as_slice(leto_r.storage());
    for i in 0..n {
        for j in 0..n {
            let got = got_r[i * n + j];
            let expected = expected_r[i * n + j];
            let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (got - expected).abs() <= tolerance,
                "blocked QR R[{i},{j}]: got {got}, expected {expected}"
            );
        }
    }

    // Least-squares solve must match leto-ops.
    let rhs_host: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
    let rhs = dev.upload(&rhs_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([m], rhs_host).unwrap();
    let solution = gpu_qr.solve_least_squares(&dev, &rhs).unwrap();
    let expected_solution = leto_qr.solve_least_squares(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; n];
    dev.download(&solution, &mut got).unwrap();
    let expected = leto::Storage::as_slice(expected_solution.storage());
    for i in 0..n {
        assert!(
            (got[i] - expected[i]).abs() <= 1e-3,
            "blocked QR solve x[{i}] = {} expected {}",
            got[i],
            expected[i]
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_identity_yields_identity_r() {
    let Some(dev) = device("blocked_qr_identity_yields_identity_r") else {
        return;
    };
    use hephaestus_cuda::{qr_decompose_blocked, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (2, 2));

    let mut got_r = vec![0.0f32; 4];
    dev.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
    let r_ref = leto_qr.r();
    let expected_r = leto::Storage::as_slice(r_ref.storage());
    for i in 0..4 {
        let tolerance = 8.0 * f32::EPSILON * expected_r[i].abs().max(1.0);
        assert!(
            (got_r[i] - expected_r[i]).abs() <= tolerance,
            "blocked QR R[{i}] = {} expected {}",
            got_r[i],
            expected_r[i]
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_solve_known_system_accurate() {
    let Some(dev) = device("blocked_qr_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::{qr_decompose_blocked, StridedOperand};
    use leto::Layout;

    // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]  =>  x = [1, 2]
    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_qr.solve_least_squares(&dev, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();

    // Verify residual A*x ≈ b.
    let residual_0 = 1.0 * got[0] + 0.0 * got[1] - 1.0;
    let residual_1 = 0.0 * got[0] + 1.0 * got[1] - 2.0;
    let residual_2 = 1.0 * got[0] + 1.0 * got[1] - 3.0;
    assert!(
        residual_0.abs() <= 1e-4,
        "blocked QR residual[0] = {residual_0}"
    );
    assert!(
        residual_1.abs() <= 1e-4,
        "blocked QR residual[1] = {residual_1}"
    );
    assert!(
        residual_2.abs() <= 1e-4,
        "blocked QR residual[2] = {residual_2}"
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_rejects_underdetermined() {
    let Some(dev) = device("blocked_qr_rejects_underdetermined") else {
        return;
    };
    use hephaestus_cuda::{qr_decompose_blocked, StridedOperand};
    use leto::Layout;

    let host = vec![0.0f32; 6];
    let input = dev.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = qr_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
    );
    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message }) if message.contains("m ≥ n")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_identity_yields_identity_lower() {
    let Some(dev) = device("blocked_cholesky_identity_yields_identity_lower") else {
        return;
    };
    use hephaestus_cuda::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_chol = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_chol.n(), 2);
    assert_eq!(gpu_chol.det(), leto_chol.det());
    assert_eq!(gpu_chol.det(), 1.0);

    let mut got_lower = vec![0.0f32; 4];
    dev.download(gpu_chol.lower(), &mut got_lower).unwrap();
    assert_eq!(got_lower, vec![1.0f32, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_spd_reconstruction_matches_original() {
    let Some(dev) = device("blocked_cholesky_spd_reconstruction_matches_original") else {
        return;
    };
    use hephaestus_cuda::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    // 66×66 SPD matrix exercises the block boundary.
    let n = 66usize;
    let mut matrix_host = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            matrix_host[row * n + col] = if row == col {
                n as f32 + 4.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host.clone()).unwrap();
    let leto_chol = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    // Reconstruct A' = L * L^T and verify against original.
    let mut got_lower = vec![0.0f32; n * n];
    dev.download(gpu_chol.lower(), &mut got_lower).unwrap();
    let expected_lower = leto::Storage::as_slice(leto_chol.lower().storage());
    for (index, (&got, &expected)) in got_lower.iter().zip(expected_lower.iter()).enumerate() {
        let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "blocked Cholesky L mismatch at {index}: got {got}, expected {expected}"
        );
    }

    for row in 0..n {
        for col in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += got_lower[row * n + k] * got_lower[col * n + k];
            }
            let expected = matrix_host[row * n + col];
            let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (sum - expected).abs() <= tolerance,
                "blocked Cholesky reconstruction [{row},{col}]: got {sum}, expected {expected}"
            );
        }
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_solve_known_system_accurate() {
    let Some(dev) = device("blocked_cholesky_solve_known_system_accurate") else {
        return;
    };
    use hephaestus_cuda::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    // A = [[4, 2], [2, 3]], b = [8, 7]  =>  x = [1.25, 1.5]
    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![8.0f32, 7.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let rhs = dev.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_chol.solve(&dev, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    dev.download(&solution, &mut got).unwrap();
    assert!(
        (got[0] - 1.25f32).abs() <= 1e-5,
        "x[0] = {} expected 1.25",
        got[0]
    );
    assert!(
        (got[1] - 1.5f32).abs() <= 1e-5,
        "x[1] = {} expected 1.5",
        got[1]
    );

    // Verify residual A*x ≈ b.
    let ax0 = 4.0 * got[0] + 2.0 * got[1];
    let ax1 = 2.0 * got[0] + 3.0 * got[1];
    assert!((ax0 - 8.0).abs() <= 1e-4, "residual[0] = {ax0}");
    assert!((ax1 - 7.0).abs() <= 1e-4, "residual[1] = {ax1}");
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_rejects_singular_matrix() {
    let Some(dev) = device("blocked_cholesky_rejects_singular_matrix") else {
        return;
    };
    use hephaestus_cuda::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose_blocked(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(
        result.is_err(),
        "singular matrix must be rejected by blocked Cholesky"
    );
}

// ────────────────────────────────────────────────────────────────────────
// Telemetry and Decomposition Contract Helper Functions
// ────────────────────────────────────────────────────────────────────────

#[cfg(feature = "decomposition")]
fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "got {actual}, expected {expected}, tolerance {tolerance}"
    );
}

fn assert_close_slice(got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        let tolerance = abs_tol.max(rel_tol * expected.abs().max(1.0));
        assert!(
            (got - expected).abs() <= tolerance,
            "slice mismatch at {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
}

#[cfg(feature = "decomposition")]
fn reconstruct_svd(
    u: &[f32],
    singular_values: &[f32],
    v: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let rank = singular_values.len();
    let mut reconstructed = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut value = 0.0f32;
            for component in 0..rank {
                value += u[row * rank + component]
                    * singular_values[component]
                    * v[col * rank + component];
            }
            reconstructed[row * cols + col] = value;
        }
    }
    reconstructed
}

#[cfg(feature = "decomposition")]
fn matmul_host(
    lhs: &[f32],
    lhs_rows: usize,
    shared: usize,
    rhs: &[f32],
    rhs_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; lhs_rows * rhs_cols];
    for row in 0..lhs_rows {
        for col in 0..rhs_cols {
            let mut value = 0.0f32;
            for k in 0..shared {
                value += lhs[row * shared + k] * rhs[k * rhs_cols + col];
            }
            out[row * rhs_cols + col] = value;
        }
    }
    out
}

#[cfg(feature = "decomposition")]
fn transpose_host(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = matrix[row * cols + col];
        }
    }
    out
}

#[cfg(feature = "decomposition")]
fn assert_orthogonal_host(matrix: &[f32], n: usize, tolerance: f32) {
    let transposed = transpose_host(matrix, n, n);
    let gram = matmul_host(&transposed, n, n, matrix, n);
    for row in 0..n {
        for col in 0..n {
            assert_close(
                gram[row * n + col],
                if row == col { 1.0 } else { 0.0 },
                tolerance,
            );
        }
    }
}

#[cfg(feature = "decomposition")]
fn sort_complex(values: &mut [num_complex::Complex<f32>]) {
    values.sort_by(|lhs, rhs| {
        lhs.re
            .total_cmp(&rhs.re)
            .then_with(|| lhs.im.total_cmp(&rhs.im))
    });
}

#[cfg(feature = "decomposition")]
fn assert_complex_spectrum_close(
    actual: &[num_complex::Complex<f32>],
    expected: &[num_complex::Complex<f32>],
    tolerance: f32,
) {
    assert_eq!(actual.len(), expected.len());
    let mut actual = actual.to_vec();
    let mut expected = expected.to_vec();
    sort_complex(&mut actual);
    sort_complex(&mut expected);
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(actual.re, expected.re, tolerance);
        assert_close(actual.im, expected.im, tolerance);
        assert!(
            ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt() <= tolerance,
            "complex spectrum mismatch at {index}: got {actual:?}, expected {expected:?}, tolerance {tolerance}"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────
// Decomposition Contract Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(feature = "decomposition")]
#[test]
fn symmetric_eigen_jacobi_matches_leto_reference() {
    let Some(dev) = device("symmetric_eigen_jacobi_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![
        4.0f32, 1.0, 0.5, 0.25, 1.0, 3.0, 0.25, 0.125, 0.5, 0.25, 2.0, 0.0625, 0.25, 0.125, 0.0625,
        1.5,
    ];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([4, 4]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([4, 4], matrix_host).unwrap();
    let leto_eigen = leto_ops::symmetric_eigen_jacobi(&leto_matrix.view()).unwrap();

    let gpu_eigen = symmetric_eigen_jacobi(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_eigen.n(), 4);
    assert_eq!(gpu_eigen.inner().eigenvalues, leto_eigen.eigenvalues);

    let mut got_values = vec![0.0f32; 4];
    dev.download(gpu_eigen.eigenvalues(), &mut got_values)
        .unwrap();
    assert_eq!(got_values, leto_eigen.eigenvalues);

    let mut got_vectors = vec![0.0f32; 16];
    dev.download(gpu_eigen.eigenvectors(), &mut got_vectors)
        .unwrap();
    assert_eq!(
        got_vectors,
        leto::Storage::as_slice(leto_eigen.eigenvectors.storage())
    );

    let values_only = symmetric_eigenvalues_jacobi(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let leto_values_only = leto_ops::symmetric_eigenvalues_jacobi(&leto_matrix.view()).unwrap();
    let mut got_values_only = vec![0.0f32; 4];
    dev.download(&values_only, &mut got_values_only).unwrap();
    assert_eq!(got_values_only, leto_values_only);
}

#[cfg(feature = "decomposition")]
#[test]
fn symmetric_eigen_jacobi_rejects_non_symmetric_input() {
    let Some(dev) = device("symmetric_eigen_jacobi_rejects_non_symmetric_input") else {
        return;
    };
    use hephaestus_cuda::{symmetric_eigen_jacobi, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 0.0, 1.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = symmetric_eigen_jacobi(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("not symmetric")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn eigenvalues_match_closed_form_diagonal() {
    let Some(dev) = device("eigenvalues_match_closed_form_diagonal") else {
        return;
    };
    use hephaestus_cuda::{eigenvalues, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![2.0f32, 0.0, 0.0, 3.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let eigen = eigenvalues(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![num_complex::Complex::new(0.0f32, 0.0); 2];
    dev.download(&eigen, &mut got).unwrap();
    got.sort_by(|lhs, rhs| lhs.re.total_cmp(&rhs.re));

    let expected = [
        num_complex::Complex::new(2.0f32, 0.0),
        num_complex::Complex::new(3.0f32, 0.0),
    ];
    for (index, (&actual, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "general eigenvalue mismatch at {index}: got {actual:?}, expected {expected:?}"
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn eigenvalues_matches_leto_reference() {
    let Some(dev) = device("eigenvalues_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{eigenvalues, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![0.0f32, 1.0, -2.0, 3.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let eigen = eigenvalues(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![num_complex::Complex::new(0.0f32, 0.0); 2];
    dev.download(&eigen, &mut got).unwrap();

    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let expected = leto_ops::eigenvalues(&leto_matrix.view()).unwrap();

    assert_eq!(got.len(), expected.len());
    for i in 0..got.len() {
        assert!(
            (got[i].re - expected[i].re).abs() < 1e-5,
            "real part mismatch at {i}: got {}, expected {}",
            got[i].re,
            expected[i].re
        );
        assert!(
            (got[i].im - expected[i].im).abs() < 1e-5,
            "imag part mismatch at {i}: got {}, expected {}",
            got[i].im,
            expected[i].im
        );
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn singular_values_match_closed_form_diagonal() {
    let Some(dev) = device("singular_values_match_closed_form_diagonal") else {
        return;
    };
    use hephaestus_cuda::{singular_values, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![3.0f32, 0.0, 0.0, 0.0, 2.0, 0.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let values = singular_values(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 2];
    dev.download(&values, &mut got).unwrap();
    assert_eq!(got.len(), 2);
    assert_close(got[0], 3.0, 1.0e-5);
    assert_close(got[1], 2.0, 1.0e-5);
}

#[cfg(feature = "decomposition")]
#[test]
fn svd_decompose_reconstructs_leto_reference() {
    let Some(dev) = device("svd_decompose_reconstructs_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{svd_decompose, StridedOperand};
    use leto::Layout;

    let rows = 4usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_svd.shape(), (rows, cols));
    let leto_matrix = leto::Array::from_shape_vec([rows, cols], matrix_host.clone()).unwrap();
    let leto_svd = leto_ops::svd_decompose(&leto_matrix.view()).unwrap();

    let rank = leto_svd.singular_values.len();
    let mut got_singular = vec![0.0f32; rank];
    let mut got_u = vec![0.0f32; rows * rank];
    let mut got_v = vec![0.0f32; cols * rank];
    dev.download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();
    dev.download(gpu_svd.u(), &mut got_u).unwrap();
    dev.download(gpu_svd.v(), &mut got_v).unwrap();

    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-5);
    }

    let reconstructed = reconstruct_svd(&got_u, &got_singular, &got_v, rows, cols);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn svd_rank_revealing_accepts_rank_deficient_matrix() {
    let Some(dev) = device("svd_rank_revealing_accepts_rank_deficient_matrix") else {
        return;
    };
    use hephaestus_cuda::{svd_rank_revealing, StridedOperand};
    use leto::Layout;

    let rows = 3usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_rank_revealing(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let leto_matrix = leto::Array::from_shape_vec([rows, cols], matrix_host).unwrap();
    let leto_svd = leto_ops::svd_rank_revealing(&leto_matrix.view()).unwrap();
    let rank = leto_svd.singular_values.len();
    let mut got_singular = vec![0.0f32; rank];
    dev.download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();

    assert_eq!(rank, 2);
    assert!(got_singular[0] >= got_singular[1]);
    assert_close(got_singular[1], 0.0, 1.0e-5);
    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn bidiagonalize_reconstructs_and_preserves_singular_values() {
    let Some(dev) = device("bidiagonalize_reconstructs_and_preserves_singular_values") else {
        return;
    };
    use hephaestus_cuda::{bidiagonalize, singular_values, StridedOperand};
    use leto::Layout;

    let rows = 4usize;
    let cols = 3usize;
    let matrix_host = vec![
        4.0f32, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
    ];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_bd = bidiagonalize(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_bd.shape(), (rows, cols));
    let mut u = vec![0.0f32; rows * rows];
    let mut b = vec![0.0f32; rows * cols];
    let mut v = vec![0.0f32; cols * cols];
    dev.download(gpu_bd.u_buffer(), &mut u).unwrap();
    dev.download(gpu_bd.b_buffer(), &mut b).unwrap();
    dev.download(gpu_bd.v_buffer(), &mut v).unwrap();

    assert_orthogonal_host(&u, rows, 1.0e-4);
    assert_orthogonal_host(&v, cols, 1.0e-4);
    for row in 0..rows {
        for col in 0..cols {
            if col < row || col > row + 1 {
                assert_close(b[row * cols + col], 0.0, 1.0e-4);
            }
        }
    }

    let ub = matmul_host(&u, rows, rows, &b, cols);
    let vt = transpose_host(&v, cols, cols);
    let reconstructed = matmul_host(&ub, rows, cols, &vt, cols);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-3);
    }

    let b_buffer = dev.upload(&b).unwrap();
    let b_layout = Layout::c_contiguous([rows, cols]).unwrap();
    let sv_b = singular_values(
        &dev,
        StridedOperand {
            buffer: &b_buffer,
            layout: &b_layout,
        },
    )
    .unwrap();
    let sv_a = singular_values(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_b = vec![0.0f32; cols];
    let mut got_a = vec![0.0f32; cols];
    dev.download(&sv_b, &mut got_b).unwrap();
    dev.download(&sv_a, &mut got_a).unwrap();
    for (actual, expected) in got_b.iter().zip(got_a.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn bidiagonalize_rejects_wide_matrix() {
    let Some(dev) = device("bidiagonalize_rejects_wide_matrix") else {
        return;
    };
    use hephaestus_cuda::{bidiagonalize, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = bidiagonalize(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );

    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Bidiagonalization requires")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn schur_reconstructs_quasi_triangular_and_preserves_spectrum() {
    let Some(dev) = device("schur_reconstructs_quasi_triangular_and_preserves_spectrum") else {
        return;
    };
    use hephaestus_cuda::{eigenvalues, schur, StridedOperand};
    use leto::Layout;

    let n = 3usize;
    let matrix_host = vec![1.0f32, -3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 5.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_schur = schur(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_schur.n(), n);
    let mut q = vec![0.0f32; n * n];
    let mut t = vec![0.0f32; n * n];
    dev.download(gpu_schur.q_buffer(), &mut q).unwrap();
    dev.download(gpu_schur.t_buffer(), &mut t).unwrap();

    assert_orthogonal_host(&q, n, 1.0e-4);
    for row in 0..n {
        for col in 0..n {
            if row > col + 1 {
                assert_close(t[row * n + col], 0.0, 1.0e-4);
            }
        }
    }
    for block in 0..(n - 1) {
        if t[(block + 1) * n + block].abs() > 1.0e-4 {
            let aa = t[block * n + block];
            let bb = t[block * n + block + 1];
            let cc = t[(block + 1) * n + block];
            let dd = t[(block + 1) * n + block + 1];
            let discriminant = (aa - dd) * (aa - dd) + 4.0 * bb * cc;
            assert!(
                discriminant <= 1.0e-4,
                "real Schur 2x2 block must encode a complex pair, discriminant {discriminant}"
            );
        }
    }

    let qt = matmul_host(&q, n, n, &t, n);
    let q_transposed = transpose_host(&q, n, n);
    let reconstructed = matmul_host(&qt, n, n, &q_transposed, n);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-3);
    }

    let t_buffer = dev.upload(&t).unwrap();
    let t_values = eigenvalues(
        &dev,
        StridedOperand {
            buffer: &t_buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let a_values = eigenvalues(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_t = vec![num_complex::Complex::new(0.0f32, 0.0); n];
    let mut got_a = vec![num_complex::Complex::new(0.0f32, 0.0); n];
    dev.download(&t_values, &mut got_t).unwrap();
    dev.download(&a_values, &mut got_a).unwrap();
    assert_complex_spectrum_close(&got_t, &got_a, 1.0e-4);
}

#[cfg(feature = "decomposition")]
#[test]
fn schur_rejects_rectangular_matrix() {
    let Some(dev) = device("schur_rejects_rectangular_matrix") else {
        return;
    };
    use hephaestus_cuda::{schur, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = schur(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );

    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Schur decomposition requires square matrix")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn hessenberg_reconstructs_and_preserves_similarity_invariants() {
    let Some(dev) = device("hessenberg_reconstructs_and_preserves_similarity_invariants") else {
        return;
    };
    use hephaestus_cuda::{hessenberg, norm_l2, trace, StridedOperand};
    use leto::Layout;

    let n = 4usize;
    let matrix_host = vec![
        4.0f32, 5.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0,
    ];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_hessenberg = hessenberg(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_hessenberg.n(), n);
    let mut q = vec![0.0f32; n * n];
    let mut h = vec![0.0f32; n * n];
    dev.download(gpu_hessenberg.q_buffer(), &mut q).unwrap();
    dev.download(gpu_hessenberg.h_buffer(), &mut h).unwrap();

    assert_orthogonal_host(&q, n, 1.0e-4);
    for row in 0..n {
        for col in 0..n {
            if row > col + 1 {
                assert_close(h[row * n + col], 0.0, 1.0e-4);
            }
        }
    }

    let qh = matmul_host(&q, n, n, &h, n);
    let q_transposed = transpose_host(&q, n, n);
    let reconstructed = matmul_host(&qh, n, n, &q_transposed, n);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-3);
    }

    let h_buffer = dev.upload(&h).unwrap();
    let h_trace = trace(
        &dev,
        StridedOperand {
            buffer: &h_buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let a_trace = trace(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_h_trace = vec![0.0f32; 1];
    let mut got_a_trace = vec![0.0f32; 1];
    dev.download(&h_trace, &mut got_h_trace).unwrap();
    dev.download(&a_trace, &mut got_a_trace).unwrap();
    assert_close(got_h_trace[0], got_a_trace[0], 1.0e-4);

    let h_norm = norm_l2(
        &dev,
        StridedOperand {
            buffer: &h_buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let a_norm = norm_l2(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_h_norm = vec![0.0f32; 1];
    let mut got_a_norm = vec![0.0f32; 1];
    dev.download(&h_norm, &mut got_h_norm).unwrap();
    dev.download(&a_norm, &mut got_a_norm).unwrap();
    assert_close(got_h_norm[0], got_a_norm[0], 1.0e-3);
}

#[cfg(feature = "decomposition")]
#[test]
fn hessenberg_rejects_rectangular_matrix() {
    let Some(dev) = device("hessenberg_rejects_rectangular_matrix") else {
        return;
    };
    use hephaestus_cuda::{hessenberg, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = hessenberg(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );

    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Hessenberg requires square matrix")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn col_piv_qr_matches_leto_reference() {
    let Some(dev) = device("col_piv_qr_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{col_piv_qr, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], matrix_host).unwrap();
    let leto_decomp = leto_ops::col_piv_qr(&leto_matrix.view()).unwrap();

    let gpu_decomp = col_piv_qr(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_decomp.rank(), leto_decomp.rank());
    assert_eq!(gpu_decomp.permutation(), leto_decomp.permutation());

    let mut q = vec![0.0f32; 9];
    let mut r = vec![0.0f32; 9];
    dev.download(gpu_decomp.q(), &mut q).unwrap();
    dev.download(gpu_decomp.r(), &mut r).unwrap();

    let q_view = leto_decomp.q();
    let r_view = leto_decomp.r();
    let expected_q = leto::Storage::as_slice(q_view.storage());
    let expected_r = leto::Storage::as_slice(r_view.storage());

    assert_close_slice(&q, expected_q, 1e-4, 0.0);
    assert_close_slice(&r, expected_r, 1e-4, 0.0);
}

#[cfg(feature = "decomposition")]
#[test]
fn full_piv_lu_matches_leto_reference() {
    let Some(dev) = device("full_piv_lu_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{full_piv_lu, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_decomp = leto_ops::full_piv_lu(&leto_matrix.view()).unwrap();

    let gpu_decomp = full_piv_lu(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_decomp.n(), 2);
    assert_eq!(gpu_decomp.rank(), leto_decomp.rank());
    assert_close(gpu_decomp.det(), leto_decomp.det(), 1e-5);
    assert_eq!(gpu_decomp.row_permutation(), leto_decomp.row_permutation());
    assert_eq!(gpu_decomp.col_permutation(), leto_decomp.col_permutation());

    let mut lu = vec![0.0f32; 4];
    dev.download(gpu_decomp.lu_buffer(), &mut lu).unwrap();
    assert_close_slice(&lu, leto_decomp.lu_factors(), 1e-4, 0.0);

    let rhs_host = vec![5.0f32, 11.0];
    let rhs = dev.upload(&rhs_host).unwrap();
    let sol = gpu_decomp.solve(&dev, &rhs).unwrap();
    let mut got_sol = vec![0.0f32; 2];
    dev.download(&sol, &mut got_sol).unwrap();

    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let expected_sol = leto_decomp.solve(&leto_rhs.view()).unwrap();
    assert_close_slice(
        &got_sol,
        leto::Storage::as_slice(expected_sol.storage()),
        1e-4,
        0.0,
    );

    let inv = gpu_decomp.inv(&dev).unwrap();
    let mut got_inv = vec![0.0f32; 4];
    dev.download(&inv, &mut got_inv).unwrap();
    let expected_inv = leto_decomp.inv().unwrap();
    assert_close_slice(
        &got_inv,
        leto::Storage::as_slice(expected_inv.storage()),
        1e-4,
        0.0,
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn udu_decompose_matches_leto_reference() {
    let Some(dev) = device("udu_decompose_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{udu_decompose, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![4.0f32, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], matrix_host).unwrap();
    let leto_decomp = leto_ops::udu_decompose(&leto_matrix.view()).unwrap();

    let gpu_decomp = udu_decompose(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_decomp.n(), 3);

    let mut u = vec![0.0f32; 9];
    let mut d = vec![0.0f32; 3];
    dev.download(gpu_decomp.u_buffer(), &mut u).unwrap();
    dev.download(gpu_decomp.d_buffer(), &mut d).unwrap();

    let u_view = leto_decomp.u();
    let expected_u = leto::Storage::as_slice(u_view.storage());
    assert_close_slice(&u, expected_u, 1e-4, 0.0);
    assert_close_slice(&d, leto_decomp.diagonal(), 1e-4, 0.0);
}

#[cfg(feature = "decomposition")]
#[test]
fn bunch_kaufman_matches_leto_reference() {
    let Some(dev) = device("bunch_kaufman_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{bunch_kaufman, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0];
    let matrix = dev.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], matrix_host).unwrap();
    let leto_decomp = leto_ops::bunch_kaufman(&leto_matrix.view()).unwrap();

    let gpu_decomp = bunch_kaufman(
        &dev,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_decomp.n(), 3);
    assert_eq!(gpu_decomp.permutation(), leto_decomp.permutation());

    let mut l = vec![0.0f32; 9];
    let mut d = vec![0.0f32; 9];
    dev.download(gpu_decomp.l_buffer(), &mut l).unwrap();
    dev.download(gpu_decomp.d_buffer(), &mut d).unwrap();

    let l_view = leto_decomp.l();
    let d_view = leto_decomp.d();
    let expected_l = leto::Storage::as_slice(l_view.storage());
    let expected_d = leto::Storage::as_slice(d_view.storage());
    assert_close_slice(&l, expected_l, 1e-4, 0.0);
    assert_close_slice(&d, expected_d, 1e-4, 0.0);
}

#[test]
fn test_cuda_uniform_and_normal_with_seed() {
    let Some(dev) = device("test_cuda_uniform_and_normal_with_seed") else {
        return;
    };
    use hephaestus_cuda::{normal_with_seed, uniform_with_seed};

    let shape = [1000];
    let low = -2.0f32;
    let high = 5.0f32;
    let u_buf = uniform_with_seed(&dev, shape, low, high, 42).unwrap();
    let mut got_u = vec![0.0f32; 1000];
    dev.download(&u_buf, &mut got_u).unwrap();

    // Verify determinism & range
    let u_buf_2 = uniform_with_seed(&dev, shape, low, high, 42).unwrap();
    let mut got_u_2 = vec![0.0f32; 1000];
    dev.download(&u_buf_2, &mut got_u_2).unwrap();
    assert_eq!(got_u, got_u_2);

    for &val in &got_u {
        assert!(val >= low && val < high, "value out of bounds: {val}");
    }

    let n_buf = normal_with_seed(&dev, shape, 0.0f32, 1.0f32, 42).unwrap();
    let mut got_n = vec![0.0f32; 1000];
    dev.download(&n_buf, &mut got_n).unwrap();
    assert!(got_n.iter().any(|&val| val != 0.0));
}

#[test]
fn test_cuda_sparse_matrix_spmv_spmm() {
    let Some(dev) = device("test_cuda_sparse_matrix_spmv_spmm") else {
        return;
    };
    use hephaestus_cuda::{spmm, spmv, GpuCsrMatrix, StridedOperand};
    use leto::Layout;

    // Create a 3x3 diagonal-ish matrix:
    // [ 2.0  0.0 -1.0 ]
    // [ 0.0  3.0  0.0 ]
    // [ 0.0  0.0  4.0 ]
    let dense_host = vec![2.0f32, 0.0, -1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let cpu_csr = leto_ops::CsrMatrix::from_dense(&leto::ArrayView2::new(layout, &dense_host));

    let gpu_csr = GpuCsrMatrix::from_cpu(&dev, &cpu_csr).unwrap();
    assert_eq!(gpu_csr.shape(), (3, 3));
    assert_eq!(gpu_csr.nnz(), 4);

    // Round-trip back to CPU
    let cpu_csr_2 = gpu_csr.to_cpu(&dev).unwrap();
    assert_eq!(cpu_csr, cpu_csr_2);

    // SpMV: y = A * x, x = [1.0, 2.0, 3.0]
    // Expected y = [ 2*1 - 3, 3*2, 4*3 ] = [ -1.0, 6.0, 12.0 ]
    let x_host = vec![1.0f32, 2.0, 3.0];
    let x_buf = dev.upload(&x_host).unwrap();
    let y_buf = spmv(&dev, &gpu_csr, &x_buf).unwrap();
    let mut got_y = vec![0.0f32; 3];
    dev.download(&y_buf, &mut got_y).unwrap();
    assert_close_slice(&got_y, &[-1.0, 6.0, 12.0], 1.0e-4, 0.0);

    // SpMM: C = A * B, B = [ 1.0  2.0 ]
    //                      [ 3.0  4.0 ]
    //                      [ 5.0  6.0 ]
    // Expected C = [ 2*1 - 5, 2*2 - 6 ] = [ -3.0, -2.0 ]
    //              [ 3*3,     3*4     ]   [  9.0, 12.0 ]
    //              [ 4*5,     4*6     ]   [ 20.0, 24.0 ]
    let b_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_buf = dev.upload(&b_host).unwrap();
    let b_layout = Layout::c_contiguous([3, 2]).unwrap();
    let b_op = StridedOperand {
        buffer: &b_buf,
        layout: &b_layout,
    };
    let c_buf = spmm(&dev, &gpu_csr, &b_op).unwrap();
    let mut got_c = vec![0.0f32; 6];
    dev.download(&c_buf, &mut got_c).unwrap();
    assert_close_slice(&got_c, &[-3.0, -2.0, 9.0, 12.0, 20.0, 24.0], 1.0e-4, 0.0);
}

/// Shared adversarial-layout driver: every non-dense view (transposed,
/// offset, broadcast/zero-stride) must be rejected by the blocked entry
/// points with the typed dense-operand error BEFORE any device copy. The
/// broadcast case is the memory-safety case: its validated storage extent
/// (4 elements here) is smaller than rows*cols, so the former raw
/// whole-matrix copy would have read past the allocation.
#[cfg(feature = "decomposition")]
fn assert_blocked_rejects_non_dense<F, O>(dev: &CudaDevice, entry: F, label: &str)
where
    F: Fn(&CudaDevice, StridedOperand<'_, f32, 2>) -> hephaestus_core::Result<O>,
{
    // 16 elements backing dense 4x4 views; 4 elements backing the broadcast.
    let dense_host: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
    let dense_buf = dev.upload(&dense_host).unwrap();
    let small_host = [1.0f32, 2.0, 3.0, 4.0];
    let small_buf = dev.upload(&small_host).unwrap();

    let transposed = Layout::new([4, 4], [1, 4], 0);
    let offset = Layout::new([3, 3], [4, 1], 5);
    let broadcast = Layout::new([4, 4], [0, 1], 0);

    for (name, layout, buffer) in [
        ("transposed", &transposed, &dense_buf),
        ("offset", &offset, &dense_buf),
        ("broadcast", &broadcast, &small_buf),
    ] {
        let result = entry(dev, StridedOperand { buffer, layout });
        match result {
            Err(HephaestusError::DispatchFailed { message }) => {
                assert!(
                    message.contains("dense C-contiguous"),
                    "{label}/{name}: rejection must name the dense-operand                      contract, got: {message}"
                );
            }
            Err(other) => {
                panic!("{label}/{name}: expected DispatchFailed dense-operand error, got {other:?}")
            }
            Ok(_) => panic!("{label}/{name}: non-dense operand must be rejected"),
        }
    }
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_cholesky_rejects_non_dense_operands() {
    let Some(dev) = device("blocked_cholesky_rejects_non_dense_operands") else {
        return;
    };
    use hephaestus_cuda::cholesky_decompose_blocked;
    assert_blocked_rejects_non_dense(&dev, cholesky_decompose_blocked, "cholesky");
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_rejects_non_dense_operands() {
    let Some(dev) = device("blocked_lu_rejects_non_dense_operands") else {
        return;
    };
    use hephaestus_cuda::lu_decompose_blocked;
    assert_blocked_rejects_non_dense(&dev, lu_decompose_blocked, "LU");
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_rejects_non_dense_operands() {
    let Some(dev) = device("blocked_qr_rejects_non_dense_operands") else {
        return;
    };
    use hephaestus_cuda::qr_decompose_blocked;
    assert_blocked_rejects_non_dense(&dev, qr_decompose_blocked, "QR");
}
