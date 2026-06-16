//! Contract tests for the CUDA `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without the `cuda` feature or without a CUDA device,
//! [`CudaDevice::try_default`] returns `Err` and each test skips.

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};
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
