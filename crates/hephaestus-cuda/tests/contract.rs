#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Contract tests for the CUDA `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without the `cuda` feature or without a CUDA device,
//! [`CudaDevice::try_default`] returns `Err` and each test skips.
//! Hardware CI sets `HEPHAESTUS_CUDA_REQUIRE_DEVICE=1` so acquisition failure
//! fails the lane instead of being reported as device evidence.

use hephaestus_core::{
    BlockWidth, ComputeDevice, ComputeDeviceCapabilities, DenseVectorOps, DeviceBuffer,
    DeviceFeature, HephaestusError, Result,
};
use hephaestus_cuda::{
    AbsOp, AddOp, CudaDevice, CudaVectorOps, CumSumOp, EluGradOp, EluOp, ExpOp, GeluTanhGradOp,
    GeluTanhOp, MaxOp, MinOp, MishGradOp, MishOp, MulOp, NegOp, ProdOp, RecipOp, SiluGradOp,
    SiluOp, SoftplusGradOp, SoftplusOp, SqrtOp, StridedOperand, SubOp, SumOp, batched_matmul,
    batched_matmul_into, binary_elementwise, binary_elementwise_into, cumprod, cumprod_into, det,
    dot, kron, matexp, matmul, matmul_into, matpow, matrix_rank, matrix_rank_with_tolerance,
    norm_l1, norm_l2, norm_max, pinv, prepare_dot, prepare_max_axis_into, prepare_mean_axis_into,
    prepare_min_axis_into, prepare_norm_l2, prepare_reduction, prepare_reduction_with_width,
    prepare_sum_axis_into, prod_axis, prod_axis_into, reduce_axis, reduction, reduction_with_width,
    scalar_elementwise, scalar_elementwise_into, scan_axis, submit_prepared_axis_reduction_batch,
    submit_prepared_reduction_batch, suffix_prod, suffix_prod_into, suffix_sum, suffix_sum_into,
    trace, unary_elementwise, unary_elementwise_into,
};
use leto::Layout;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct AlternateIdentity(i32);

impl hephaestus_core::DialectScalar<hephaestus_core::CudaC> for AlternateIdentity {
    const TYPE_TOKEN: &'static str = "int";
}

impl hephaestus_cuda::MatrixIdentityScalar for AlternateIdentity {
    const ZERO: Self = Self(-7);
    const ONE: Self = Self(9);
}

/// Acquire a device, or `None` to skip (no `cuda` feature / no GPU).
fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_some() {
                panic!("CUDA device required for {test}, but acquisition failed: {error}");
            }
            eprintln!("skip {test}: CUDA device unavailable ({error})");
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
fn device_local_copy_preserves_values_and_rejects_mismatch() {
    let Some(dev) = device("device_local_copy_preserves_values_and_rejects_mismatch") else {
        return;
    };
    let host: Vec<f32> = (0..1027).map(|index| index as f32 * 0.25 - 17.0).collect();
    let source = dev.upload(&host).expect("upload source");
    let destination = dev
        .alloc_zeroed::<f32>(host.len())
        .expect("allocate destination");

    dev.copy_buffer(&source, &destination)
        .expect("device-local copy");
    let mut copied = vec![0.0_f32; host.len()];
    dev.download(&destination, &mut copied)
        .expect("download copied values");
    assert_eq!(copied, host);

    let empty_source = dev.upload::<f32>(&[]).expect("upload empty source");
    let empty_destination = dev
        .alloc_zeroed::<f32>(0)
        .expect("allocate empty destination");
    dev.copy_buffer(&empty_source, &empty_destination)
        .expect("empty device-local copy");

    let short = dev
        .alloc_zeroed::<f32>(host.len() - 1)
        .expect("allocate short destination");
    assert_length_mismatch(dev.copy_buffer(&source, &short), source.len(), short.len());
}

#[test]
fn uninitialized_allocation_is_fully_overwritten_before_read() {
    let Some(dev) = device("uninitialized_allocation_is_fully_overwritten_before_read") else {
        return;
    };
    let expected = [1.0_f32, -2.5, 3.25, 8.0];
    let buffer = dev
        .alloc_uninitialized::<f32>(expected.len())
        .expect("uninitialized allocation");
    dev.write_buffer(&buffer, &expected)
        .expect("full-buffer initialization");
    let mut actual = [0.0_f32; 4];
    dev.download(&buffer, &mut actual).expect("download");
    assert_eq!(actual, expected);
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
fn elementwise_activation_markers_match_cpu_reference() {
    let Some(device) = device("elementwise_activation_markers_match_cpu_reference") else {
        return;
    };
    let host = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let input = device.upload(&host).unwrap();
    let gelu_scale = 0.797_884_6_f32;
    let gelu_cubic = 0.044715_f32;

    macro_rules! check {
        ($label:literal, $operation:ty, $expected:expr) => {{
            let expected = $expected;
            let output = unary_elementwise::<$operation, f32>(&device, &input).unwrap();
            let mut actual = [0.0_f32; 5];
            device.download(&output, &mut actual).unwrap();
            for (index, (&got, &reference)) in actual.iter().zip(expected.iter()).enumerate() {
                let tolerance = f32::EPSILON * 512.0 * reference.abs().max(1.0);
                assert!(
                    (got - reference).abs() <= tolerance,
                    "{}[{index}] got {got}, expected {reference}, tolerance {tolerance}",
                    $label
                );
            }
        }};
    }

    check!(
        "gelu_tanh",
        GeluTanhOp,
        host.iter()
            .map(|&x| 0.5 * x * (1.0 + (gelu_scale * (x + gelu_cubic * x * x * x)).tanh()))
            .collect::<Vec<_>>()
    );
    check!(
        "gelu_tanh_grad",
        GeluTanhGradOp,
        host.iter()
            .map(|&x| {
                let tanh = (gelu_scale * (x + gelu_cubic * x * x * x)).tanh();
                0.5 * (1.0 + tanh)
                    + 0.5 * x * (1.0 - tanh * tanh) * gelu_scale * (1.0 + 3.0 * gelu_cubic * x * x)
            })
            .collect::<Vec<_>>()
    );
    check!(
        "silu",
        SiluOp,
        host.iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect::<Vec<_>>()
    );
    check!(
        "silu_grad",
        SiluGradOp,
        host.iter()
            .map(|&x| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))
            })
            .collect::<Vec<_>>()
    );
    check!(
        "softplus",
        SoftplusOp,
        host.iter()
            .map(|&x| (1.0 + x.exp()).ln())
            .collect::<Vec<_>>()
    );
    check!(
        "softplus_grad",
        SoftplusGradOp,
        host.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect::<Vec<_>>()
    );
    check!(
        "mish",
        MishOp,
        host.iter()
            .map(|&x| x * (1.0 + x.exp()).ln().tanh())
            .collect::<Vec<_>>()
    );
    check!(
        "mish_grad",
        MishGradOp,
        host.iter()
            .map(|&x| {
                let softplus = (1.0 + x.exp()).ln();
                softplus.tanh() + x * (1.0 - softplus.tanh().powi(2)) / (1.0 + (-x).exp())
            })
            .collect::<Vec<_>>()
    );
    check!(
        "elu",
        EluOp,
        host.iter()
            .map(|&x| if x >= 0.0 { x } else { x.exp() - 1.0 })
            .collect::<Vec<_>>()
    );
    check!(
        "elu_grad",
        EluGradOp,
        host.iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { x.exp() })
            .collect::<Vec<_>>()
    );
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
fn prepared_reduction_reuses_device_outputs_and_batches() {
    let Some(dev) = device("prepared_reduction_reuses_device_outputs_and_batches") else {
        return;
    };

    let host: Vec<u32> = (0..1027).map(|index| (index % 17) as u32).collect();
    let expected_sum: u32 = host.iter().sum();
    let expected_min = host.iter().copied().min().expect("non-empty input");
    let expected_max = host.iter().copied().max().expect("non-empty input");
    let input = dev.upload(&host).unwrap();
    let width = BlockWidth::new(128).unwrap();

    let sum = prepare_reduction_with_width::<SumOp, _>(&dev, &input, width).unwrap();
    let output_ptr = sum.output().raw();
    let mut got = [u32::MAX];
    dev.download(sum.output(), &mut got).unwrap();
    assert_eq!(got, [0]);
    sum.dispatch().unwrap();
    dev.download(sum.output(), &mut got).unwrap();
    assert_eq!(got, [expected_sum]);
    sum.dispatch().unwrap();
    dev.download(sum.output(), &mut got).unwrap();
    assert_eq!(got, [expected_sum]);
    assert_eq!(sum.output().raw(), output_ptr);

    let min = prepare_reduction::<MinOp, _>(&dev, &input).unwrap();
    let max = prepare_reduction::<MaxOp, _>(&dev, &input).unwrap();
    submit_prepared_reduction_batch(&[&min, &max]).unwrap();
    let mut got_min = [0_u32];
    let mut got_max = [0_u32];
    dev.download(min.output(), &mut got_min).unwrap();
    dev.download(max.output(), &mut got_max).unwrap();
    assert_eq!(got_min, [expected_min]);
    assert_eq!(got_max, [expected_max]);

    let empty = dev.upload::<u32>(&[]).unwrap();
    let prepared_empty = prepare_reduction::<SumOp, _>(&dev, &empty).unwrap();
    prepared_empty.dispatch().unwrap();
    let mut got_empty = [u32::MAX];
    dev.download(prepared_empty.output(), &mut got_empty)
        .unwrap();
    assert_eq!(got_empty, [0]);

    let invalid_width = BlockWidth::new(192).unwrap();
    assert_dispatch_message(
        prepare_reduction_with_width::<SumOp, _>(&dev, &input, invalid_width),
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
fn linalg_batched_matmul_matches_cpu_reference() {
    let Some(dev) = device("linalg_batched_matmul_matches_cpu_reference") else {
        return;
    };

    // Batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]]     = [[19,22],[43,50]]
    // Batch 1: [[9,10],[11,12]] @ [[13,14],[15,16]] = [[267,286],[323,346]]
    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0];
    let b_host = vec![5.0f32, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0];
    let expected = vec![19.0f32, 22.0, 43.0, 50.0, 267.0, 286.0, 323.0, 346.0];

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(8).unwrap();

    let a_layout = Layout::c_contiguous([2, 2, 2]).unwrap();
    let b_layout = Layout::c_contiguous([2, 2, 2]).unwrap();
    let out_layout = Layout::c_contiguous([2, 2, 2]).unwrap();

    let allocated = batched_matmul(
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
    let mut allocated_values = vec![0.0f32; 8];
    dev.download(&allocated, &mut allocated_values).unwrap();
    assert_eq!(allocated_values, expected);

    batched_matmul_into(
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

    let mut got = vec![0.0f32; 8];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn linalg_batched_matmul_broadcasts_single_batch_lhs() {
    let Some(dev) = device("linalg_batched_matmul_broadcasts_single_batch_lhs") else {
        return;
    };

    // lhs has batch=1 and broadcasts across rhs's batch=3:
    //   lhs = [[1,2],[3,4]]
    //   rhs0 = I        -> out0 = lhs            = [[1,2],[3,4]]
    //   rhs1 = 2*I       -> out1 = 2*lhs          = [[2,4],[6,8]]
    //   rhs2 = swap cols -> out2 = lhs @ swap     = [[2,1],[4,3]]
    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_host = vec![
        1.0f32, 0.0, 0.0, 1.0, // identity
        2.0, 0.0, 0.0, 2.0, // 2 * identity
        0.0, 1.0, 1.0, 0.0, // column swap
    ];
    let expected = vec![
        1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 2.0, 1.0, 4.0, 3.0,
    ];

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(12).unwrap();

    let a_layout = Layout::c_contiguous([1, 2, 2]).unwrap();
    let b_layout = Layout::c_contiguous([3, 2, 2]).unwrap();
    let out_layout = Layout::c_contiguous([3, 2, 2]).unwrap();

    batched_matmul_into(
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
fn prepared_map_reductions_reuse_resources_and_validate_layouts() {
    let Some(dev) = device("prepared_map_reductions_reuse_resources_and_validate_layouts") else {
        return;
    };

    let lhs = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let rhs = dev.upload(&[5.0_f32, 6.0, 7.0, 8.0]).unwrap();
    let contiguous = Layout::c_contiguous([4]).unwrap();
    let prepared_dot = prepare_dot(
        &dev,
        StridedOperand {
            buffer: &lhs,
            layout: &contiguous,
        },
        StridedOperand {
            buffer: &rhs,
            layout: &contiguous,
        },
    )
    .unwrap();
    let dot_output = prepared_dot.output() as *const _;
    prepared_dot.dispatch().unwrap();
    let mut got = [0.0_f32];
    dev.download(prepared_dot.output(), &mut got).unwrap();
    assert_eq!(got, [70.0]);
    assert_eq!(dot_output, prepared_dot.output() as *const _);

    dev.write_buffer(&lhs, &[2.0_f32, 2.0, 2.0, 2.0]).unwrap();
    prepared_dot.dispatch().unwrap();
    dev.download(prepared_dot.output(), &mut got).unwrap();
    assert_eq!(got, [52.0]);
    assert_eq!(dot_output, prepared_dot.output() as *const _);

    let reversed = Layout::try_new([4], [-1], 3).expect("valid test layout");
    let reversed_dot = prepare_dot(
        &dev,
        StridedOperand {
            buffer: &rhs,
            layout: &reversed,
        },
        StridedOperand {
            buffer: &lhs,
            layout: &contiguous,
        },
    )
    .unwrap();
    reversed_dot.dispatch().unwrap();
    dev.download(reversed_dot.output(), &mut got).unwrap();
    assert_eq!(got, [52.0]);

    let norm_input = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let transposed = Layout::try_new([2, 2], [1, 2], 0).expect("valid test layout");
    let prepared_norm = prepare_norm_l2(
        &dev,
        StridedOperand {
            buffer: &norm_input,
            layout: &transposed,
        },
    )
    .unwrap();
    let norm_output = prepared_norm.output() as *const _;
    dev.write_buffer(prepared_norm.output(), &[f32::NAN])
        .unwrap();
    prepared_norm.dispatch().unwrap();
    dev.download(prepared_norm.output(), &mut got).unwrap();
    let expected = 30.0_f32.sqrt();
    assert!((got[0] - expected).abs() <= 2.0 * f32::EPSILON * expected.max(1.0));
    assert_eq!(norm_output, prepared_norm.output() as *const _);

    dev.write_buffer(&norm_input, &[4.0_f32; 4]).unwrap();
    dev.write_buffer(prepared_norm.output(), &[f32::NAN])
        .unwrap();
    prepared_norm.dispatch().unwrap();
    dev.download(prepared_norm.output(), &mut got).unwrap();
    assert_eq!(got, [8.0]);
    assert_eq!(norm_output, prepared_norm.output() as *const _);

    let empty = dev.upload::<f32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0]).unwrap();
    let empty_dot = prepare_dot(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .unwrap();
    empty_dot.dispatch().unwrap();
    dev.download(empty_dot.output(), &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let empty_norm = prepare_norm_l2(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .unwrap();
    empty_norm.dispatch().unwrap();
    dev.download(empty_norm.output(), &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let wrong_shape = Layout::c_contiguous([3]).unwrap();
    assert!(matches!(
        prepare_dot(
            &dev,
            StridedOperand {
                buffer: &lhs,
                layout: &contiguous,
            },
            StridedOperand {
                buffer: &rhs,
                layout: &wrong_shape,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("dot product shape mismatch:")
    ));

    let invalid_layout = Layout::try_new([3], [1], 2).expect("valid test layout");
    assert!(matches!(
        prepare_norm_l2(
            &dev,
            StridedOperand {
                buffer: &lhs,
                layout: &invalid_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
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
fn linalg_fused_map_reductions_accept_reversed_views() {
    let Some(dev) = device("linalg_fused_map_reductions_accept_reversed_views") else {
        return;
    };

    let values = dev.upload(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let weights = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let reversed = Layout::try_new([4], [-1], 3).expect("valid test layout");
    let contiguous = Layout::c_contiguous([4]).unwrap();

    let reversed_operand = StridedOperand {
        buffer: &values,
        layout: &reversed,
    };
    let weights_operand = StridedOperand {
        buffer: &weights,
        layout: &contiguous,
    };

    let dot_buf = dot(&dev, reversed_operand, weights_operand).unwrap();
    let l1_buf = norm_l1(&dev, reversed_operand).unwrap();
    let l2_buf = norm_l2(&dev, reversed_operand).unwrap();
    let max_buf = norm_max(&dev, reversed_operand).unwrap();

    let mut got = [0.0_f32; 1];
    dev.download(&dot_buf, &mut got).unwrap();
    assert_eq!(got, [0.0]);
    dev.download(&l1_buf, &mut got).unwrap();
    assert_eq!(got, [10.0]);
    dev.download(&l2_buf, &mut got).unwrap();
    let expected_l2 = 30.0_f32.sqrt();
    // The four-term tree is exact for this integer square sum; the tolerance
    // allows two f32 ulps for the final square-root rounding.
    assert!((got[0] - expected_l2).abs() <= 2.0 * f32::EPSILON * expected_l2);
    dev.download(&max_buf, &mut got).unwrap();
    assert_eq!(got, [4.0]);
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
fn linalg_matpow_matches_leto_and_strided_references() {
    let Some(dev) = device("linalg_matpow_matches_leto_and_strided_references") else {
        return;
    };

    let shear_host = vec![1.0_f32, 1.0, 0.0, 1.0];
    let shear = dev.upload(&shear_host).unwrap();
    let square_layout = Layout::c_contiguous([2, 2]).unwrap();
    let shear_power = matpow(
        &dev,
        StridedOperand {
            buffer: &shear,
            layout: &square_layout,
        },
        5,
    )
    .unwrap();
    let leto_shear = leto::Array::from_shape_vec([2, 2], shear_host).unwrap();
    let expected_shear = leto_ops::matpow(&leto_shear.view(), 5).unwrap().into_vec();
    let mut got_shear = [0.0_f32; 4];
    dev.download(&shear_power, &mut got_shear).unwrap();
    assert_eq!(got_shear.as_slice(), expected_shear.as_slice());

    let strided_values = [99.0_f32, 1.0, 2.0, 3.0, 4.0];
    let strided = dev.upload(&strided_values).unwrap();
    let strided_layout = Layout::try_new([2, 2], [1, 2], 1).expect("valid test layout");
    let strided_power = matpow(
        &dev,
        StridedOperand {
            buffer: &strided,
            layout: &strided_layout,
        },
        2,
    )
    .unwrap();
    let mut got_strided = [0.0_f32; 4];
    dev.download(&strided_power, &mut got_strided).unwrap();
    assert_eq!(got_strided, [7.0, 15.0, 10.0, 22.0]);

    let identity_power = matpow(
        &dev,
        StridedOperand {
            buffer: &strided,
            layout: &strided_layout,
        },
        0,
    )
    .unwrap();
    let mut got_identity = [0.0_f32; 4];
    dev.download(&identity_power, &mut got_identity).unwrap();
    assert_eq!(got_identity, [1.0, 0.0, 0.0, 1.0]);

    let alternate = dev.upload(&[AlternateIdentity(4); 4]).unwrap();
    let alternate_power = matpow(
        &dev,
        StridedOperand {
            buffer: &alternate,
            layout: &square_layout,
        },
        0,
    )
    .unwrap();
    let mut got_alternate = [AlternateIdentity(0); 4];
    dev.download(&alternate_power, &mut got_alternate).unwrap();
    assert_eq!(
        got_alternate,
        [
            AlternateIdentity(9),
            AlternateIdentity(-7),
            AlternateIdentity(-7),
            AlternateIdentity(9),
        ]
    );

    let empty = dev.upload::<f32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0, 0]).unwrap();
    let empty_power = matpow(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        0,
    )
    .unwrap();
    assert_eq!(empty_power.len(), 0);

    let nonsquare_values = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let nonsquare_layout = Layout::c_contiguous([2, 3]).unwrap();
    assert_dispatch_message(
        matpow(
            &dev,
            StridedOperand {
                buffer: &nonsquare_values,
                layout: &nonsquare_layout,
            },
            2,
        ),
        "matpow requires a square matrix, got shape [2, 3]",
    );
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
fn linalg_matrix_functions_preserve_empty_outputs() {
    let Some(dev) = device("linalg_matrix_functions_preserve_empty_outputs") else {
        return;
    };

    let matrix = dev.upload(&[] as &[f32]).unwrap();
    let layout = Layout::c_contiguous([0, 0]).unwrap();
    let operand = StridedOperand {
        buffer: &matrix,
        layout: &layout,
    };

    assert_eq!(pinv(&dev, operand).unwrap().len(), 0);
    assert_eq!(matexp(&dev, operand).unwrap().len(), 0);
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

    let product_columns = reduce_axis::<ProdOp, f32>(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product_columns = [0.0f32; 3];
    dev.download(&product_columns, &mut got_product_columns)
        .unwrap();
    assert_eq!(got_product_columns, [4.0, 10.0, 18.0]);

    let product = prod_axis::<f32>(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product = vec![0.0f32; 2];
    dev.download(&product, &mut got_product).unwrap();
    assert_eq!(got_product, vec![6.0, 120.0]);

    let product_output = dev.alloc_zeroed::<f32>(2).unwrap();
    let product_output_layout = Layout::c_contiguous([2, 1]).unwrap();
    prod_axis_into(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        1,
        StridedOperand {
            buffer: &product_output,
            layout: &product_output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product_into = [0.0f32; 2];
    dev.download(&product_output, &mut got_product_into)
        .unwrap();
    assert_eq!(got_product_into, [6.0, 120.0]);

    let transposed_layout = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
    let transposed_product = prod_axis(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_transposed_product = [0.0f32; 3];
    dev.download(&transposed_product, &mut got_transposed_product)
        .unwrap();
    assert_eq!(got_transposed_product, [4.0, 10.0, 18.0]);

    let wrong_product_output = dev.alloc_zeroed::<f32>(6).unwrap();
    let wrong_product_layout = Layout::c_contiguous([2, 3]).unwrap();
    assert!(matches!(
        prod_axis_into(
            &dev,
            StridedOperand {
                buffer: &a,
                layout: &a_layout,
            },
            1,
            StridedOperand {
                buffer: &wrong_product_output,
                layout: &wrong_product_layout,
            },
            BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("axis reduction output shape mismatch")
    ));

    let empty = dev.upload::<f32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0, 2]).unwrap();
    let empty_product = prod_axis(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_empty_product = [0.0f32; 2];
    dev.download(&empty_product, &mut got_empty_product)
        .unwrap();
    assert_eq!(got_empty_product, [1.0, 1.0]);
}

#[test]
fn prepared_axis_reductions_reuse_plans_and_validate_contracts() {
    let Some(dev) = device("prepared_axis_reductions_reuse_plans_and_validate_contracts") else {
        return;
    };

    let host: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let input = dev.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([3, 4]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let width = BlockWidth::new(2).unwrap();

    let axis0_out = dev.alloc_zeroed::<f32>(4).unwrap();
    let axis0_layout = Layout::c_contiguous([1, 4]).unwrap();
    let prepared_sum_axis0 = prepare_sum_axis_into(
        &dev,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        width,
    )
    .unwrap();
    prepared_sum_axis0.dispatch(&dev).unwrap();
    let mut got_axis0 = [0.0f32; 4];
    dev.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);
    prepared_sum_axis0.dispatch(&dev).unwrap();
    dev.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);

    let transposed_layout = Layout::try_new([4, 3], [1, 4], 0).expect("valid test layout");
    let transposed_input = StridedOperand {
        buffer: &input,
        layout: &transposed_layout,
    };
    let axis1_out = dev.alloc_zeroed::<f32>(4).unwrap();
    let axis1_layout = Layout::c_contiguous([4, 1]).unwrap();
    let prepared_sum_axis1 = prepare_sum_axis_into(
        &dev,
        transposed_input,
        1,
        StridedOperand {
            buffer: &axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    let max_axis0_out = dev.alloc_zeroed::<f32>(3).unwrap();
    let max_axis0_layout = Layout::c_contiguous([1, 3]).unwrap();
    let prepared_max_axis0 = prepare_max_axis_into(
        &dev,
        transposed_input,
        0,
        StridedOperand {
            buffer: &max_axis0_out,
            layout: &max_axis0_layout,
        },
        width,
    )
    .unwrap();
    submit_prepared_axis_reduction_batch(&dev, &[&prepared_sum_axis1, &prepared_max_axis0])
        .unwrap();
    let mut got_axis1 = [0.0f32; 4];
    let mut got_max_axis0 = [0.0f32; 3];
    dev.download(&axis1_out, &mut got_axis1).unwrap();
    dev.download(&max_axis0_out, &mut got_max_axis0).unwrap();
    assert_eq!(got_axis1, [15.0, 18.0, 21.0, 24.0]);
    assert_eq!(got_max_axis0, [4.0, 8.0, 12.0]);

    let mean_axis1_out = dev.alloc_zeroed::<f32>(4).unwrap();
    let prepared_mean_axis1 = prepare_mean_axis_into(
        &dev,
        transposed_input,
        1,
        StridedOperand {
            buffer: &mean_axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    prepared_mean_axis1.dispatch(&dev).unwrap();
    let mut got_mean_axis1 = [0.0f32; 4];
    dev.download(&mean_axis1_out, &mut got_mean_axis1).unwrap();
    assert_eq!(got_mean_axis1, [5.0, 6.0, 7.0, 8.0]);

    let empty_input = dev.upload::<f32>(&[]).unwrap();
    let empty_input_layout = Layout::c_contiguous([3, 0]).unwrap();
    let empty_output = dev.upload(&[7.0f32; 3]).unwrap();
    let empty_output_layout = Layout::c_contiguous([3, 1]).unwrap();
    let prepared_empty_sum = prepare_sum_axis_into(
        &dev,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    )
    .unwrap();
    prepared_empty_sum.dispatch(&dev).unwrap();
    let mut got_empty = [7.0f32; 3];
    dev.download(&empty_output, &mut got_empty).unwrap();
    assert_eq!(got_empty, [0.0, 0.0, 0.0]);

    let empty_product = prod_axis::<f32>(
        &dev,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        width,
    )
    .unwrap();
    let mut got_empty_product = [0.0f32; 3];
    dev.download(&empty_product, &mut got_empty_product)
        .unwrap();
    assert_eq!(got_empty_product, [1.0, 1.0, 1.0]);

    let empty_min = prepare_min_axis_into(
        &dev,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    );
    assert!(matches!(
        empty_min,
        Err(HephaestusError::DispatchFailed { message })
            if message == "min_axis is undefined for empty axis 1"
    ));

    let alias_layout = Layout::c_contiguous([3, 1]).unwrap();
    let alias = prepare_sum_axis_into(
        &dev,
        input_operand,
        1,
        StridedOperand {
            buffer: &input,
            layout: &alias_layout,
        },
        width,
    );
    assert!(matches!(
        alias,
        Err(HephaestusError::DispatchFailed { message })
            if message == "axis reduction output buffer must not alias input buffer"
    ));

    let invalid_width = BlockWidth::new(3).unwrap();
    let invalid = prepare_sum_axis_into(
        &dev,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        invalid_width,
    );
    assert!(matches!(
        invalid,
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 3 must be a power of two"
    ));
}

#[test]
fn scan_scan_axis_matches_cpu() {
    let Some(dev) = device("scan_scan_axis_matches_cpu") else {
        return;
    };

    let host_in = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = dev.upload(&host_in).unwrap();
    let a_layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = scan_axis::<_, CumSumOp, f32>(
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
fn scan_suffix_sum_matches_leto_for_allocated_and_caller_owned_outputs() {
    let Some(dev) = device("scan_suffix_sum_matches_leto_for_allocated_and_caller_owned_outputs")
    else {
        return;
    };

    let host = vec![1i32, 2, 3, 4, 5, 6];
    let input = dev.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let expected = leto_ops::scan_axis::<leto_ops::CumSumOp, _, 2>(
        &leto::Array::from_shape_vec([2, 3], host).unwrap().view(),
        1,
        leto_ops::ScanDirection::Reverse,
    )
    .unwrap()
    .into_vec();

    let allocated = suffix_sum(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_allocated = vec![0i32; expected.len()];
    dev.download(&allocated, &mut got_allocated).unwrap();
    assert_eq!(got_allocated, expected);

    let output = dev.alloc_zeroed::<i32>(expected.len()).unwrap();
    suffix_sum_into(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
        1,
        StridedOperand {
            buffer: &output,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_into = vec![0i32; expected.len()];
    dev.download(&output, &mut got_into).unwrap();
    assert_eq!(got_into, expected);
}

#[test]
fn scan_cumprod_convenience_preserves_strided_and_empty_contract() {
    let Some(dev) = device("scan_cumprod_convenience_preserves_strided_and_empty_contract") else {
        return;
    };

    let physical = vec![1_i32, 2, 3, 4, 5, 6];
    let input = dev.upload(&physical).unwrap();
    let transposed_layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let output_layout = Layout::c_contiguous([2, 3]).unwrap();
    let output = dev.alloc_zeroed::<i32>(6).unwrap();
    cumprod_into(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = [0_i32; 6];
    dev.download(&output, &mut got).unwrap();
    assert_eq!(got, [1, 3, 15, 2, 8, 48]);

    let allocated = cumprod(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_allocated = [0_i32; 6];
    dev.download(&allocated, &mut got_allocated).unwrap();
    assert_eq!(got_allocated, [1, 3, 5, 2, 12, 30]);

    let suffix = suffix_prod(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix = [0_i32; 6];
    dev.download(&suffix, &mut got_suffix).unwrap();
    assert_eq!(got_suffix, [15, 15, 5, 48, 24, 6]);

    let suffix_into = dev.alloc_zeroed::<i32>(6).unwrap();
    suffix_prod_into(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        StridedOperand {
            buffer: &suffix_into,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_into = [0_i32; 6];
    dev.download(&suffix_into, &mut got_suffix_into).unwrap();
    assert_eq!(got_suffix_into, [2, 12, 30, 2, 4, 6]);

    let empty = dev.alloc_zeroed::<i32>(0).unwrap();
    let empty_layout = Layout::c_contiguous([2, 0]).unwrap();
    let empty_output = cumprod(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    assert_eq!(empty_output.len(), 0);

    let invalid_layout = Layout::try_new([2, 3], [1, 2], 1).expect("valid test layout");
    assert!(matches!(
        cumprod(
            &dev,
            StridedOperand {
                buffer: &input,
                layout: &invalid_layout,
            },
            1,
            BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}

#[test]
fn scan_long_line_matches_integer_reference() {
    let Some(dev) = device("scan_long_line_matches_integer_reference") else {
        return;
    };

    let cols = 513usize;
    let host: Vec<i32> = (0..2 * cols)
        .map(|index| i32::try_from(index).expect("test index fits i32") - 300)
        .collect();
    let input = dev.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, cols]).unwrap();
    let output = scan_axis::<_, CumSumOp, i32>(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
        1,
        hephaestus_cuda::ScanDirection::Forward,
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0i32; host.len()];
    dev.download(&output, &mut got).unwrap();
    let expected: Vec<i32> = host
        .chunks_exact(cols)
        .flat_map(|row| {
            let mut acc = 0i32;
            row.iter().map(move |value| {
                acc += *value;
                acc
            })
        })
        .collect();
    assert_eq!(got, expected);
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
fn blocked_cholesky_matches_leto_reference_across_block_boundary() {
    let Some(dev) = device("blocked_cholesky_matches_leto_reference_across_block_boundary") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, cholesky_decompose_blocked};
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
        // Two backward-stable Choleskys of this strictly diagonally
        // dominant fixture (κ∞ ≤ 1.01) differ elementwise by at most
        // 2·c(n)·ε·κ∞·max(|L|, 1) with c(n) ≤ n (Higham, Accuracy and
        // Stability, ch. 10); 4·n·ε keeps 2× slack over that bound.
        let tolerance = 4.0 * n as f32 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "blocked Cholesky lower mismatch at {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
    // Bitwise det equality pins provider identity: both dets come from
    // the same leto elimination on the host, so any divergence means the
    // adapter re-derived it.
    assert_eq!(gpu_cholesky.det(), leto_cholesky.det());
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

// ── Blocked decomposition differential tests ────────────────────────────

#[cfg(feature = "decomposition")]
#[test]
fn blocked_lu_matches_leto_reference() {
    let Some(dev) = device("blocked_lu_matches_leto_reference") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, lu_decompose_blocked};
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

    assert_eq!(gpu_lu.n(), leto_lu.dim());
    // Bitwise det equality pins provider identity (same leto elimination
    // on the host feeds both sides).
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
    // Two backward-stable solves of this strictly diagonally dominant
    // system (κ∞ ≤ 1.03; growth ρ ≤ 2 ⇒ c(n) ≤ 3n, Higham ch. 9) differ
    // by at most 2·c(n)·ε·κ∞·‖x‖∞; 12·n·ε·‖x‖∞ keeps ~2× slack.
    let x_inf = expected.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
    let solve_bound = 12.0 * n as f32 * f32::EPSILON * x_inf;
    for i in 0..n {
        assert!(
            (got[i] - expected[i]).abs() <= solve_bound,
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
    use hephaestus_cuda::{StridedOperand, lu_decompose_blocked};
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
    use hephaestus_cuda::{StridedOperand, lu_decompose_blocked};
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
        // Every elimination and substitution step on this fixture is
        // dyadic (pivot 4, multiplier 1/2, U₂₂ = −1/2), so both solves
        // are exact; the bound admits one reciprocal-multiply rounding
        // per step.
        assert!(
            (got[i] - expected[i]).abs() <= 4.0 * f32::EPSILON * expected[i].abs(),
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
    use hephaestus_cuda::{StridedOperand, lu_decompose_blocked};
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
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
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

    // R's lower triangle is written as zeros, never computed; ε admits
    // at most one rounded store.
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
            // Householder QR is columnwise backward stable:
            // ‖ΔR·eⱼ‖₂ ≤ c(m,n)·ε·‖aⱼ‖₂ (Higham ch. 19) with ‖aⱼ‖₂ ≤ 5.1
            // here, so 4·m·ε·max(|R|, 1) dominates the elementwise
            // difference of two stable runs on this fixture.
            let tolerance = 4.0 * m as f32 * f32::EPSILON * expected.abs().max(1.0);
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
    // For this near-consistent, well-conditioned system (κ₂ ≈ 1, so the
    // κ²·residual term of least-squares sensitivity is negligible) two
    // backward-stable solves differ by ≤ 2·c(m)·ε·‖x‖∞ with c(m) ≤ 4·m;
    // 8·m·ε·‖x‖∞ doubles that.
    let x_inf = expected.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
    let solve_bound = 8.0 * m as f32 * f32::EPSILON * x_inf;
    for i in 0..n {
        assert!(
            (got[i] - expected[i]).abs() <= solve_bound,
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
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
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
        // Identity input reaches R through at most one Householder
        // reflection over exact dyadic values: ≤ 4 roundings per entry.
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
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
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

    // The system is consistent with exact solution [1, 2]; Householder
    // QR at m = 3 spends tens of flops per entry, so the residual is
    // bounded by c·ε·‖b‖∞ with c ≲ 32.
    let residual_bound = 32.0 * f32::EPSILON * 3.0;
    let residual_0 = 1.0 * got[0] + 0.0 * got[1] - 1.0;
    let residual_1 = 0.0 * got[0] + 1.0 * got[1] - 2.0;
    let residual_2 = 1.0 * got[0] + 1.0 * got[1] - 3.0;
    assert!(
        residual_0.abs() <= residual_bound,
        "blocked QR residual[0] = {residual_0}"
    );
    assert!(
        residual_1.abs() <= residual_bound,
        "blocked QR residual[1] = {residual_1}"
    );
    assert!(
        residual_2.abs() <= residual_bound,
        "blocked QR residual[2] = {residual_2}"
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_qr_rejects_underdetermined() {
    let Some(dev) = device("blocked_qr_rejects_underdetermined") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, qr_decompose_blocked};
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
    use hephaestus_cuda::{StridedOperand, cholesky_decompose_blocked};
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
    use hephaestus_cuda::{StridedOperand, cholesky_decompose_blocked};
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
    use hephaestus_cuda::{StridedOperand, cholesky_decompose_blocked};
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
    use hephaestus_cuda::{StridedOperand, cholesky_decompose_blocked};
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
fn sort_complex(values: &mut [eunomia::Complex<f32>]) {
    values.sort_by(|lhs, rhs| {
        lhs.re
            .total_cmp(&rhs.re)
            .then_with(|| lhs.im.total_cmp(&rhs.im))
    });
}

#[cfg(feature = "decomposition")]
fn assert_complex_spectrum_close(
    actual: &[eunomia::Complex<f32>],
    expected: &[eunomia::Complex<f32>],
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
            ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt()
                <= tolerance,
            "complex spectrum mismatch at {index}: got {actual:?}, expected {expected:?}, tolerance {tolerance}"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────
// Decomposition Contract Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(feature = "decomposition")]
#[test]
fn symmetric_eigen_jacobi_rejects_non_symmetric_input() {
    let Some(dev) = device("symmetric_eigen_jacobi_rejects_non_symmetric_input") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, symmetric_eigen_jacobi};
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
    use hephaestus_cuda::{StridedOperand, eigenvalues};
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

    let mut got = vec![eunomia::Complex::new(0.0f32, 0.0); 2];
    dev.download(&eigen, &mut got).unwrap();
    got.sort_by(|lhs, rhs| lhs.re.total_cmp(&rhs.re));

    let expected = [
        eunomia::Complex::new(2.0f32, 0.0),
        eunomia::Complex::new(3.0f32, 0.0),
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
fn singular_values_match_closed_form_diagonal() {
    let Some(dev) = device("singular_values_match_closed_form_diagonal") else {
        return;
    };
    use hephaestus_cuda::{StridedOperand, singular_values};
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
    use hephaestus_cuda::{StridedOperand, svd_decompose};
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
    use hephaestus_cuda::{StridedOperand, svd_rank_revealing};
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
    let leto_svd = leto_ops::svd_decompose(&leto_matrix.view()).unwrap();
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
    use hephaestus_cuda::{StridedOperand, bidiagonalize, singular_values};
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
    use hephaestus_cuda::{StridedOperand, bidiagonalize};
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
    use hephaestus_cuda::{StridedOperand, eigenvalues, schur};
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
    let mut got_t = vec![eunomia::Complex::new(0.0f32, 0.0); n];
    let mut got_a = vec![eunomia::Complex::new(0.0f32, 0.0); n];
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
    use hephaestus_cuda::{StridedOperand, schur};
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
    use hephaestus_cuda::{StridedOperand, hessenberg, norm_l2, trace};
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
    use hephaestus_cuda::{StridedOperand, hessenberg};
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
fn blocked_pivoted_decompositions_match_ordinary_contracts() {
    let Some(dev) = device("blocked_pivoted_decompositions_match_ordinary_contracts") else {
        return;
    };
    use hephaestus_cuda::{
        StridedOperand, col_piv_qr, col_piv_qr_blocked, full_piv_lu, full_piv_lu_blocked,
    };
    use leto::Layout;

    let square_host = [2.0f32, 5.0, -2.0, 1.0, 2.0, 3.0, -2.0, 4.0, 3.0];
    let square = dev.upload(&square_host).unwrap();
    let square_layout = Layout::c_contiguous([3, 3]).unwrap();
    let square_operand = StridedOperand {
        buffer: &square,
        layout: &square_layout,
    };
    let ordinary_lu = full_piv_lu(&dev, square_operand).unwrap();
    let blocked_lu = full_piv_lu_blocked(&dev, square_operand).unwrap();
    assert_eq!(blocked_lu.rank(), ordinary_lu.rank());
    assert_eq!(blocked_lu.row_permutation(), ordinary_lu.row_permutation());
    assert_eq!(blocked_lu.col_permutation(), ordinary_lu.col_permutation());
    assert_close(blocked_lu.det(), ordinary_lu.det(), 1.0e-5);
    let mut ordinary_factors = vec![0.0f32; 9];
    let mut blocked_factors = vec![0.0f32; 9];
    dev.download(ordinary_lu.lu_buffer(), &mut ordinary_factors)
        .unwrap();
    dev.download(blocked_lu.lu_buffer(), &mut blocked_factors)
        .unwrap();
    assert_close_slice(&blocked_factors, &ordinary_factors, 1.0e-5, 0.0);

    let tall_host = [1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];
    let tall = dev.upload(&tall_host).unwrap();
    let tall_layout = Layout::c_contiguous([3, 2]).unwrap();
    let tall_operand = StridedOperand {
        buffer: &tall,
        layout: &tall_layout,
    };
    let ordinary_qr = col_piv_qr(&dev, tall_operand).unwrap();
    let blocked_qr = col_piv_qr_blocked(&dev, tall_operand).unwrap();
    assert_eq!(blocked_qr.rank(), ordinary_qr.rank());
    assert_eq!(blocked_qr.permutation(), ordinary_qr.permutation());
    let mut ordinary_q = vec![0.0f32; 9];
    let mut blocked_q = vec![0.0f32; 9];
    let mut ordinary_r = vec![0.0f32; 6];
    let mut blocked_r = vec![0.0f32; 6];
    dev.download(ordinary_qr.q(), &mut ordinary_q).unwrap();
    dev.download(blocked_qr.q(), &mut blocked_q).unwrap();
    dev.download(ordinary_qr.r(), &mut ordinary_r).unwrap();
    dev.download(blocked_qr.r(), &mut blocked_r).unwrap();
    assert_close_slice(&blocked_q, &ordinary_q, 1.0e-5, 0.0);
    assert_close_slice(&blocked_r, &ordinary_r, 1.0e-5, 0.0);
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
    use hephaestus_cuda::{GpuCsrMatrix, StridedOperand, spmm, spmv};
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

    let empty_row_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 4.0],
        vec![0, 2, 2],
        vec![0, 2, 2, 3],
        3,
        3,
    )
    .unwrap();
    let empty_row_gpu = GpuCsrMatrix::from_cpu(&dev, &empty_row_csr).unwrap();
    let empty_row_y = spmv(&dev, &empty_row_gpu, &x_buf).unwrap();
    let mut got_empty_row_y = [0.0_f32; 3];
    dev.download(&empty_row_y, &mut got_empty_row_y).unwrap();
    assert_eq!(got_empty_row_y, [-1.0, 0.0, 12.0]);

    let empty_row_c = spmm(&dev, &empty_row_gpu, &b_op).unwrap();
    let mut got_empty_row_c = [0.0_f32; 6];
    dev.download(&empty_row_c, &mut got_empty_row_c).unwrap();
    assert_eq!(got_empty_row_c, [-3.0, -2.0, 0.0, 0.0, 20.0, 24.0]);
}

#[test]
fn test_cuda_prepared_sparse_dispatch_matches_reference() {
    let Some(dev) = device("test_cuda_prepared_sparse_dispatch_matches_reference") else {
        return;
    };
    use hephaestus_cuda::{
        GpuCsrMatrix, PreparedSparseDispatch, StridedOperand, prepare_spmm, prepare_spmv,
        prepare_spmv_many, submit_prepared_sparse_batch,
    };

    let cpu_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .unwrap();
    let gpu_csr = GpuCsrMatrix::from_cpu(&dev, &cpu_csr).unwrap();
    let x = dev.upload(&[1.0_f32, 2.0, 3.0]).unwrap();
    let y = dev.upload(&[77.0_f32; 3]).unwrap();
    let prepared_spmv = prepare_spmv(&dev, &gpu_csr, &x, &y).unwrap();
    prepared_spmv.dispatch().unwrap();
    prepared_spmv.dispatch().unwrap();

    let b = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b_layout = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
    let b_operand = StridedOperand {
        buffer: &b,
        layout: &b_layout,
    };
    let mut c = dev.upload(&[88.0_f32; 6]).unwrap();
    let prepared_spmm = prepare_spmm(&dev, &gpu_csr, &b_operand, &mut c).unwrap();
    let mut many = dev.upload(&[99.0_f32; 6]).unwrap();
    let prepared_many = prepare_spmv_many(&dev, &gpu_csr, &b_operand, &mut many).unwrap();
    submit_prepared_sparse_batch(&[
        PreparedSparseDispatch::Spmv(&prepared_spmv),
        PreparedSparseDispatch::Spmm(&prepared_spmm),
        PreparedSparseDispatch::Spmm(&prepared_many),
    ])
    .unwrap();

    let mut got_y = [0.0_f32; 3];
    dev.download(&y, &mut got_y).unwrap();
    assert_close_slice(&got_y, &[-1.0, 6.0, 12.0], 1.0e-4, 0.0);
    let mut got_c = [0.0_f32; 6];
    dev.download(&c, &mut got_c).unwrap();
    assert_close_slice(&got_c, &[-1.0, 2.0, 6.0, 15.0, 12.0, 24.0], 1.0e-4, 0.0);
    let mut got_many = [0.0_f32; 6];
    dev.download(&many, &mut got_many).unwrap();
    assert_eq!(got_many, got_c);

    let wrong_x = dev.upload(&[1.0_f32, 2.0]).unwrap();
    let wrong_output = dev.upload(&[0.0_f32; 3]).unwrap();
    assert_length_mismatch(prepare_spmv(&dev, &gpu_csr, &wrong_x, &wrong_output), 3, 2);
    let bad_layout = Layout::try_new([3, 2], [2, 1], 5).expect("valid test layout");
    let bad_operand = StridedOperand {
        buffer: &b,
        layout: &bad_layout,
    };
    let mut bad_output = dev.upload(&[0.0_f32; 6]).unwrap();
    match prepare_spmm(&dev, &gpu_csr, &bad_operand, &mut bad_output) {
        Err(HephaestusError::DispatchFailed { message }) => {
            assert!(
                message.contains("layout rejected"),
                "unexpected error: {message}"
            );
        }
        Err(error) => panic!("expected invalid layout rejection, got {error:?}"),
        Ok(_) => panic!("expected invalid layout rejection, got success"),
    }
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

    let transposed = Layout::try_new([4, 4], [1, 4], 0).expect("valid test layout");
    let offset = Layout::try_new([3, 3], [4, 1], 5).expect("valid test layout");
    let broadcast = Layout::try_new([4, 4], [0, 1], 0).expect("valid test layout");

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

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_reject_non_dense_operands() {
    let Some(dev) = device("blocked_pivoted_decompositions_reject_non_dense_operands") else {
        return;
    };
    assert_blocked_rejects_non_dense(
        &dev,
        hephaestus_cuda::full_piv_lu_blocked,
        "complete-pivoted LU",
    );
    assert_blocked_rejects_non_dense(
        &dev,
        hephaestus_cuda::col_piv_qr_blocked,
        "column-pivoted QR",
    );
}

#[cfg(feature = "decomposition")]
#[test]
fn empty_decompositions_preserve_shapes_and_identities() {
    use hephaestus_cuda::{
        bidiagonalize, col_piv_qr, full_piv_lu, hessenberg, qr_decompose, qr_decompose_blocked,
    };

    let Some(dev) = device("empty_decompositions_preserve_shapes_and_identities") else {
        return;
    };
    let empty = dev.alloc_zeroed::<f32>(0).unwrap();
    let tall_layout = Layout::c_contiguous([3, 0]).unwrap();
    let tall = StridedOperand {
        buffer: &empty,
        layout: &tall_layout,
    };

    let bidiagonal = bidiagonalize(&dev, tall).unwrap();
    assert_eq!(bidiagonal.shape(), (3, 0));
    assert_eq!(bidiagonal.u_buffer().len(), 9);
    assert_eq!(bidiagonal.b_buffer().len(), 0);
    assert_eq!(bidiagonal.v_buffer().len(), 0);
    let expected_identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut identity = vec![0.0; 9];
    dev.download(bidiagonal.u_buffer(), &mut identity).unwrap();
    assert_eq!(identity, expected_identity);

    let pivoted = col_piv_qr(&dev, tall).unwrap();
    assert_eq!(pivoted.rank(), 0);
    assert_eq!(pivoted.permutation(), &[] as &[usize]);
    assert_eq!(pivoted.q().len(), 9);
    assert_eq!(pivoted.r().len(), 0);
    identity.fill(0.0);
    dev.download(pivoted.q(), &mut identity).unwrap();
    assert_eq!(identity, expected_identity);

    for qr in [
        qr_decompose(&dev, tall).unwrap(),
        qr_decompose_blocked(&dev, tall).unwrap(),
    ] {
        assert_eq!(qr.shape(), (3, 0));
        assert_eq!(qr.r_buffer().len(), 0);
        assert_eq!(qr.inner().shape(), (3, 0));
        assert_eq!(
            leto::Storage::as_slice(qr.inner().q().storage()),
            expected_identity
        );
        assert_eq!(qr.inner().r().shape(), [3, 0]);
    }

    let square_layout = Layout::c_contiguous([0, 0]).unwrap();
    let square = StridedOperand {
        buffer: &empty,
        layout: &square_layout,
    };
    let lu = full_piv_lu(&dev, square).unwrap();
    assert_eq!(lu.n(), 0);
    assert_eq!(lu.rank(), 0);
    assert_eq!(lu.det(), 1.0);
    assert_eq!(lu.row_permutation(), &[] as &[usize]);
    assert_eq!(lu.col_permutation(), &[] as &[usize]);
    assert_eq!(lu.lu_buffer().len(), 0);

    let h = hessenberg(&dev, square).unwrap();
    assert_eq!(h.n(), 0);
    assert_eq!(h.q_buffer().len(), 0);
    assert_eq!(h.h_buffer().len(), 0);
}

fn assert_vector_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * want.abs().max(1.0),
            "index {index}: got {got}, expected {want}"
        );
    }
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

/// The regression guard for the defect itself.
///
/// Neither value clause below discriminates it: the old field was a free
/// reading *stored at acquisition*, so it was stale rather than moving, and
/// `buffer_limit_is_stable_across_allocations_and_free_memory_tracks_them`
/// passes on the defective code. `free <= max_buffer_size` also holds when
/// the limit *is* a free reading. What distinguishes the two is which half of
/// `cuMemGetInfo_v2` reaches `DeviceLimits`, so the source is the oracle —
/// and this clause needs no device, so it guards on every runner.
#[test]
fn the_buffer_limit_is_built_from_total_device_memory_not_the_free_reading() {
    let source = include_str!("../src/infrastructure/device.rs");
    let body = source
        .split_once("fn query_device_limits(")
        .map(|(_, tail)| tail)
        .and_then(|tail| {
            tail.split_once(
                "
fn ",
            )
        })
        .map(|(body, _)| body)
        .expect("query_device_limits must be present");

    assert!(
        body.contains("let (_, total_bytes) = current_memory_info()?;"),
        "query_device_limits must take the total half of current_memory_info"
    );
    assert!(
        !body.contains("free_bytes"),
        "no free-memory reading may reach DeviceLimits; it is a runtime query"
    );
}

/// `max_buffer_size` is the device capacity — it does not move when memory is
/// allocated — while the free-memory query does.
#[test]
fn buffer_limit_is_stable_across_allocations_and_free_memory_tracks_them() {
    let Some(dev) = device("buffer_limit_is_stable_across_allocations") else {
        return;
    };
    let limits_before = dev.device_limits();
    let free_before = dev.free_memory_bytes().expect("free-memory query");
    // 64 MiB: large enough that the driver must back it with new device pages
    // rather than serve it from a pool the previous tests already touched.
    let _held = dev
        .alloc_zeroed::<f32>(16 << 20)
        .expect("64 MiB device buffer");
    let free_after = dev.free_memory_bytes().expect("free-memory query");
    assert_eq!(
        dev.device_limits(),
        limits_before,
        "the limit moved with an allocation"
    );
    assert!(
        free_after < free_before,
        "free memory did not fall across a 64 MiB allocation: {free_before} -> {free_after}"
    );
}
