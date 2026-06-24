//! Differential contract tests: wgpu dispatch vs CPU reference.
//!
//! Tests acquire a real adapter; on hosts without one (headless CI without
//! GPU/lavapipe) they skip with a message rather than fabricate a pass.

use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{
    binary_elementwise, binary_elementwise_into, cumsum_into, matrix_rank,
    matrix_rank_with_tolerance, max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis,
    min_axis_into, reduction, reduction_with_width, scalar_elementwise, scalar_elementwise_into,
    sum_axis, sum_axis_into, unary_elementwise, unary_elementwise_into, AbsOp, AddOp,
    ComputeDevice, DeviceBuffer, ExpOp, HephaestusError, MaxOp, MinOp, MulOp, NegOp, RecipOp,
    SqrtOp, SubOp, SumOp, WgpuDevice,
};

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: std::sync::OnceLock<Option<WgpuDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(
            || match WgpuDevice::try_default("hephaestus-contract-test") {
                Ok(device) => Some(device),
                Err(e) => {
                    eprintln!("skipping wgpu contract test: {e}");
                    None
                }
            },
        )
        .clone()
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
fn assert_length_mismatch<T>(
    result: hephaestus_wgpu::Result<T>,
    host_len: usize,
    device_len: usize,
) {
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

fn assert_dispatch_message<T>(result: hephaestus_wgpu::Result<T>, expected: &str) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => assert_eq!(message, expected),
        Err(error) => panic!("expected dispatch failure {expected:?}, got {error:?}"),
        Ok(_) => panic!("expected dispatch failure {expected:?}, got success"),
    }
}

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

fn assert_complex_spectra_close(
    got: &[num_complex::Complex<f32>],
    expected: &[num_complex::Complex<f32>],
    abs_tol: f32,
    rel_tol: f32,
) {
    assert_eq!(got.len(), expected.len());
    let mut used = vec![false; got.len()];
    for (expected_index, expected) in expected.iter().enumerate() {
        let match_index = got.iter().enumerate().position(|(got_index, actual)| {
            if used[got_index] {
                return false;
            }
            let re_tolerance = abs_tol.max(rel_tol * expected.re.abs().max(1.0));
            let im_tolerance = abs_tol.max(rel_tol * expected.im.abs().max(1.0));
            (actual.re - expected.re).abs() <= re_tolerance
                && (actual.im - expected.im).abs() <= im_tolerance
        });
        match match_index {
            Some(index) => used[index] = true,
            None => {
                panic!("no eigenvalue matches oracle[{expected_index}] = {expected:?}; got {got:?}")
            }
        }
    }
}

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

fn transpose_host(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = matrix[row * cols + col];
        }
    }
    out
}

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

fn sort_complex(values: &mut [num_complex::Complex<f32>]) {
    values.sort_by(|lhs, rhs| {
        lhs.re
            .total_cmp(&rhs.re)
            .then_with(|| lhs.im.total_cmp(&rhs.im))
    });
}

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
            (actual - expected).norm() <= tolerance,
            "complex spectrum mismatch at {index}: got {actual:?}, expected {expected:?}, tolerance {tolerance}"
        );
    }
}

fn packed_lu_product(packed: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut value = 0.0f32;
            for k in 0..n {
                let l = if row > k {
                    packed[row * n + k]
                } else if row == k {
                    1.0
                } else {
                    0.0
                };
                let u = if k <= col { packed[k * n + col] } else { 0.0 };
                value += l * u;
            }
            out[row * n + col] = value;
        }
    }
    out
}

fn ldl_transpose_product(l: &[f32], d: &[f32], n: usize) -> Vec<f32> {
    let ld = matmul_host(l, n, n, d, n);
    let lt = transpose_host(l, n, n);
    matmul_host(&ld, n, n, &lt, n)
}

fn udu_transpose_product(u: &[f32], d: &[f32], n: usize) -> Vec<f32> {
    let mut ud = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            ud[row * n + col] = u[row * n + col] * d[col];
        }
    }
    let ut = transpose_host(u, n, n);
    matmul_host(&ud, n, n, &ut, n)
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
fn test_placement_aware_allocation() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use themis::{MemoryTier, PlacementHint};

    let host: Vec<f32> = (0..128).map(|i| i as f32 * 0.25 - 7.0).collect();

    // HostPinned buffers are persistently host-mapped staging buffers
    // (`hephaestus-mnemosyne-staging`): the upload variant is MAP_WRITE|COPY_SRC,
    // the zeroed variant MAP_READ|COPY_DST. Because the buffer stays mapped, it
    // cannot be a `copy_buffer_to_buffer` source/destination (wgpu rejects a
    // queue submit touching a mapped buffer), so `download`/compute dispatch are
    // not its access path — the host reads/writes through the mapped pointer.
    // The publicly verifiable placement contract for this tier is therefore the
    // recorded tier and element length on both constructors.
    let pinned = PlacementHint::Tier(MemoryTier::HostPinned);
    let buf_pinned = device.upload_with_hint(&host, pinned).unwrap();
    assert_eq!(buf_pinned.len(), 128);
    assert_eq!(buf_pinned.tier(), MemoryTier::HostPinned);

    let zeroed_pinned = device.alloc_zeroed_with_hint::<f32>(128, pinned).unwrap();
    assert_eq!(zeroed_pinned.len(), 128);
    assert_eq!(zeroed_pinned.tier(), MemoryTier::HostPinned);

    // Dram tier: a device-local STORAGE buffer. The hint changes placement,
    // never values, so an upload must round-trip identically to the default
    // path and a zeroed allocation must be genuinely zero-initialized.
    let dram = PlacementHint::Tier(MemoryTier::Dram);
    let buf_dram = device.upload_with_hint(&host, dram).unwrap();
    assert_eq!(buf_dram.tier(), MemoryTier::Dram);
    let mut dram_out = vec![0.0f32; 128];
    device.download(&buf_dram, &mut dram_out).unwrap();
    assert_eq!(dram_out, host, "Dram upload must preserve data");

    let zeroed_dram = device.alloc_zeroed_with_hint::<f32>(128, dram).unwrap();
    let mut zeroed_out = vec![1.0f32; 128];
    device.download(&zeroed_dram, &mut zeroed_out).unwrap();
    assert_eq!(
        zeroed_out,
        vec![0.0f32; 128],
        "Dram alloc_zeroed must zero-initialize"
    );

    // Default (non-hinted) allocation lands on Device and round-trips data.
    let buf_default = device.upload(&host).unwrap();
    assert_eq!(buf_default.tier(), MemoryTier::Device);
    let mut default_out = vec![0.0f32; 128];
    device.download(&buf_default, &mut default_out).unwrap();
    assert_eq!(default_out, host, "Device upload must preserve data");
}

#[test]
fn download_rejects_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let buffer = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let mut out = vec![0.0f32; 2];
    assert_length_mismatch(device.download(&buffer, &mut out), 2, 3);
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
    assert_length_mismatch(binary_elementwise::<AddOp, f32>(&device, &a, &b), 2, 3);
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
    assert_length_mismatch(
        binary_elementwise_into::<AddOp, f32>(&device, &a, &b, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        unary_elementwise_into::<NegOp, f32>(&device, &a, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        scalar_elementwise_into::<AddOp, f32>(&device, &a, 1.0, &short, width),
        short.len(),
        a.len(),
    );
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
fn reduction_width_is_part_of_dispatch_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let host: Vec<u32> = (0..1027).collect();
    let expected: u32 = host.iter().sum();
    let input = device.upload(&host).unwrap();

    let narrow = BlockWidth::new(128).unwrap();
    let out_narrow = reduction_with_width::<SumOp, u32>(&device, &input, narrow).unwrap();
    let mut got_narrow = vec![0u32; 1];
    device.download(&out_narrow, &mut got_narrow).unwrap();
    assert_eq!(got_narrow[0], expected);

    let non_power = BlockWidth::new(192).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&device, &input, non_power),
        "reduction block width 192 must be a power of two",
    );

    let empty = device.upload::<u32>(&[]).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&device, &empty, non_power),
        "reduction block width 192 must be a power of two",
    );

    let singleton = device.upload(&[13u32]).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&device, &singleton, non_power),
        "reduction block width 192 must be a power of two",
    );
}

#[test]
fn axis_reductions_match_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = device.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([2, 3]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let leto_input = leto::Array::from_shape_vec([2, 3], host).unwrap();

    let out_axis0 = device.alloc_zeroed::<f32>(3).unwrap();
    let out_axis0_layout = Layout::c_contiguous([1, 3]).unwrap();
    sum_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &out_axis0,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_axis0 = leto_ops::sum_axis(&leto_input.view(), 0)
        .unwrap()
        .into_vec();
    let mut got_axis0 = vec![0.0f32; 3];
    device.download(&out_axis0, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, expected_axis0);

    let allocated_axis0 = sum_axis(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_axis0 = vec![0.0f32; 3];
    device
        .download(&allocated_axis0, &mut got_allocated_axis0)
        .unwrap();
    assert_eq!(got_allocated_axis0, expected_axis0);

    let out_axis1 = device.alloc_zeroed::<f32>(2).unwrap();
    let out_axis1_layout = Layout::c_contiguous([2, 1]).unwrap();
    sum_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &out_axis1,
            layout: &out_axis1_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_axis1 = leto_ops::sum_axis(&leto_input.view(), 1)
        .unwrap()
        .into_vec();
    let mut got_axis1 = vec![0.0f32; 2];
    device.download(&out_axis1, &mut got_axis1).unwrap();
    assert_eq!(got_axis1, expected_axis1);

    let min_axis0 = device.alloc_zeroed::<f32>(3).unwrap();
    min_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &min_axis0,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_min_axis0 = leto_ops::min_axis(&leto_input.view(), 0)
        .unwrap()
        .into_vec();
    let mut got_min_axis0 = vec![0.0f32; 3];
    device.download(&min_axis0, &mut got_min_axis0).unwrap();
    assert_eq!(got_min_axis0, expected_min_axis0);

    let allocated_min_axis0 = min_axis(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_min_axis0 = vec![0.0f32; 3];
    device
        .download(&allocated_min_axis0, &mut got_allocated_min_axis0)
        .unwrap();
    assert_eq!(got_allocated_min_axis0, expected_min_axis0);

    let max_axis1 = device.alloc_zeroed::<f32>(2).unwrap();
    max_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &max_axis1,
            layout: &out_axis1_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_max_axis1 = leto_ops::max_axis(&leto_input.view(), 1)
        .unwrap()
        .into_vec();
    let mut got_max_axis1 = vec![0.0f32; 2];
    device.download(&max_axis1, &mut got_max_axis1).unwrap();
    assert_eq!(got_max_axis1, expected_max_axis1);

    let allocated_max_axis1 = max_axis(&device, input_operand, 1, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_max_axis1 = vec![0.0f32; 2];
    device
        .download(&allocated_max_axis1, &mut got_allocated_max_axis1)
        .unwrap();
    assert_eq!(got_allocated_max_axis1, expected_max_axis1);

    let mean_axis0 = device.alloc_zeroed::<f32>(3).unwrap();
    mean_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &mean_axis0,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_mean_axis0 = leto_ops::mean_axis(&leto_input.view(), 0)
        .unwrap()
        .into_vec();
    let mut got_mean_axis0 = vec![0.0f32; 3];
    device.download(&mean_axis0, &mut got_mean_axis0).unwrap();
    assert_eq!(got_mean_axis0, expected_mean_axis0);

    let allocated_mean_axis0 = mean_axis(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_mean_axis0 = vec![0.0f32; 3];
    device
        .download(&allocated_mean_axis0, &mut got_allocated_mean_axis0)
        .unwrap();
    assert_eq!(got_allocated_mean_axis0, expected_mean_axis0);
}

#[test]
fn axis_scans_match_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{
        cumsum, scan_axis, scan_axis_into, CumProdOp, ScanDirection, StridedOperand,
    };
    use leto::Layout;

    let host = vec![1i32, 2, 3, 4, 5, 6];
    let input = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &layout,
    };
    let leto_input = leto::Array::from_shape_vec([2, 3], host).unwrap();

    let cumsum_axis1 = device.alloc_zeroed::<i32>(6).unwrap();
    cumsum_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &cumsum_axis1,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_cumsum_axis1 = leto_ops::cumsum(&leto_input.view(), 1).unwrap().into_vec();
    let mut got_cumsum_axis1 = vec![0i32; 6];
    device
        .download(&cumsum_axis1, &mut got_cumsum_axis1)
        .unwrap();
    assert_eq!(got_cumsum_axis1, expected_cumsum_axis1);

    let cumsum_allocated = cumsum(&device, input_operand, 1, BlockWidth::DEFAULT).unwrap();
    let mut got_cumsum_allocated = vec![0i32; 6];
    device
        .download(&cumsum_allocated, &mut got_cumsum_allocated)
        .unwrap();
    assert_eq!(got_cumsum_allocated, expected_cumsum_axis1);

    let cumprod_reverse = device.alloc_zeroed::<i32>(6).unwrap();
    scan_axis_into::<CumProdOp, i32>(
        &device,
        input_operand,
        0,
        ScanDirection::Reverse,
        StridedOperand {
            buffer: &cumprod_reverse,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_cumprod_reverse = leto_ops::scan_axis::<leto_ops::CumProdOp, _, 2>(
        &leto_input.view(),
        0,
        leto_ops::ScanDirection::Reverse,
    )
    .unwrap()
    .into_vec();
    let mut got_cumprod_reverse = vec![0i32; 6];
    device
        .download(&cumprod_reverse, &mut got_cumprod_reverse)
        .unwrap();
    assert_eq!(got_cumprod_reverse, expected_cumprod_reverse);

    let cumprod_reverse_allocated = scan_axis::<CumProdOp, i32>(
        &device,
        input_operand,
        0,
        ScanDirection::Reverse,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_cumprod_reverse_allocated = vec![0i32; 6];
    device
        .download(
            &cumprod_reverse_allocated,
            &mut got_cumprod_reverse_allocated,
        )
        .unwrap();
    assert_eq!(got_cumprod_reverse_allocated, expected_cumprod_reverse);
}

#[test]
fn acquisition_reports_themis_topology_from_adapter() {
    eprintln!("DEBUG: Test started");
    let Some(device) = device_or_skip() else {
        eprintln!("DEBUG: Device skipped");
        return;
    };
    eprintln!("DEBUG: Device acquired");
    let topology = device
        .topology()
        .expect("acquisition path must capture a topology snapshot");
    eprintln!("DEBUG: Topology retrieved");

    // Verify reported fields have reasonable defaults/values
    assert!(topology.warp_width() == 0 || topology.warp_width().is_power_of_two());
    eprintln!("DEBUG: Warp width checked");

    // Unreported-by-wgpu capacities must be zero, never fabricated.
    assert_eq!(topology.compute_units(), 0);
    assert_eq!(topology.registers_per_unit(), 0);
    assert_eq!(topology.shared_mem_per_unit_bytes(), 0);
    eprintln!("DEBUG: Capacities checked");

    // The Arc-wrapping constructor has no adapter and reports none.
    let wrapped = WgpuDevice::new(device.device().clone(), device.queue().clone());
    eprintln!("DEBUG: Wrapped device created");
    assert_eq!(
        wrapped.topology().map(|topology| topology.compute_units()),
        None
    );
    eprintln!("DEBUG: Test finished successfully");
}

#[test]
fn linalg_matmul_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{matmul, matmul_into, StridedOperand};
    use leto::Layout;

    // Multiply two f32 matrices: shape [3, 2] x [2, 4] -> [3, 4]
    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_host = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let expected = vec![
        29.0f32, 32.0, 35.0, 38.0, 65.0, 72.0, 79.0, 86.0, 101.0, 112.0, 123.0, 134.0,
    ];

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(12).unwrap();

    let a_layout = Layout::c_contiguous([3, 2]).unwrap();
    let b_layout = Layout::c_contiguous([2, 4]).unwrap();
    let out_layout = Layout::c_contiguous([3, 4]).unwrap();

    matmul_into(
        &device,
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
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);

    let allocated = matmul(
        &device,
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
    let mut allocated_got = vec![0.0f32; 12];
    device.download(&allocated, &mut allocated_got).unwrap();
    assert_eq!(allocated_got, expected);
}

#[test]
fn linalg_batched_matmul_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{batched_matmul, batched_matmul_into, StridedOperand};
    use leto::Layout;

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_host = vec![2.0f32, 0.5, 1.0, 3.0];
    let expected = vec![4.0f32, 6.5, 10.0, 13.5, 16.0, 20.5, 22.0, 27.5];

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(expected.len()).unwrap();
    let a_layout = Layout::c_contiguous([2, 2, 2]).unwrap();
    let b_layout = Layout::c_contiguous([1, 2, 2]).unwrap();
    let out_layout = Layout::c_contiguous([2, 2, 2]).unwrap();

    batched_matmul_into(
        &device,
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

    let mut got = vec![0.0f32; expected.len()];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);

    let allocated = batched_matmul(
        &device,
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
    let mut allocated_got = vec![0.0f32; expected.len()];
    device.download(&allocated, &mut allocated_got).unwrap();
    assert_eq!(allocated_got, expected);
}

#[test]
fn linalg_kron_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{kron, kron_into, StridedOperand};
    use leto::Layout;

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_host = vec![7.0f32, 8.0, 9.0, 10.0];
    let leto_a = leto::Array::from_shape_vec([2, 3], a_host.clone()).unwrap();
    let leto_b = leto::Array::from_shape_vec([2, 2], b_host.clone()).unwrap();
    let expected = leto_ops::kron(&leto_a.view(), &leto_b.view())
        .unwrap()
        .into_vec();

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(expected.len()).unwrap();
    let a_layout = Layout::c_contiguous([2, 3]).unwrap();
    let b_layout = Layout::c_contiguous([2, 2]).unwrap();
    let out_layout = Layout::c_contiguous([4, 6]).unwrap();

    kron_into(
        &device,
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

    let mut got = vec![0.0f32; expected.len()];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);

    let allocated = kron(
        &device,
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
    let mut allocated_got = vec![0.0f32; expected.len()];
    device.download(&allocated, &mut allocated_got).unwrap();
    assert_eq!(allocated_got, expected);
}

#[test]
fn linalg_matpow_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{matpow, StridedOperand};
    use leto::Layout;

    let shear_host = vec![1.0f32, 1.0, 0.0, 1.0];
    let shear = device.upload(&shear_host).unwrap();
    let shear_layout = Layout::c_contiguous([2, 2]).unwrap();
    let shear_pow = matpow(
        &device,
        StridedOperand {
            buffer: &shear,
            layout: &shear_layout,
        },
        5,
    )
    .unwrap();
    let leto_shear = leto::Array::from_shape_vec([2, 2], shear_host).unwrap();
    let expected_shear = leto_ops::matpow(&leto_shear.view(), 5).unwrap().into_vec();
    let mut got_shear = vec![0.0f32; 4];
    device.download(&shear_pow, &mut got_shear).unwrap();
    assert_eq!(got_shear, expected_shear);

    let diagonal_host = vec![2i32, 0, 0, 3];
    let diagonal = device.upload(&diagonal_host).unwrap();
    let diagonal_pow = matpow(
        &device,
        StridedOperand {
            buffer: &diagonal,
            layout: &shear_layout,
        },
        0,
    )
    .unwrap();
    let mut got_diagonal = vec![0i32; 4];
    device.download(&diagonal_pow, &mut got_diagonal).unwrap();
    assert_eq!(got_diagonal, vec![1, 0, 0, 1]);
}

#[test]
fn linalg_matpow_rejects_non_square() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{matpow, StridedOperand};
    use leto::Layout;

    let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    assert_dispatch_message(
        matpow(
            &device,
            StridedOperand {
                buffer: &input,
                layout: &layout,
            },
            2,
        ),
        "matpow requires a square matrix, got shape [2, 3]",
    );
}

#[test]
fn linalg_dot_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{dot, StridedOperand};
    use leto::Layout;

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
    let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0; // 70.0

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();

    let a_layout = Layout::c_contiguous([4]).unwrap();
    let b_layout = Layout::c_contiguous([4]).unwrap();

    let out_buf = dot(
        &device,
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
    device.download(&out_buf, &mut got).unwrap();
    assert_eq!(got[0], expected);
}

#[test]
fn linalg_trace_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{trace, StridedOperand};
    use leto::Layout;

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected = 1.0 + 5.0 + 9.0; // 15.0

    let a = device.upload(&a_host).unwrap();
    let a_layout = Layout::c_contiguous([3, 3]).unwrap();

    let out_buf = trace(
        &device,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
    )
    .unwrap();

    let mut got = [0.0f32; 1];
    device.download(&out_buf, &mut got).unwrap();
    assert_eq!(got[0], expected);
}

#[test]
fn linalg_matrix_rank_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    let full_rank_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let deficient_host = vec![1.0f32, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
    let zero_host = vec![0.0f32; 6];
    let tolerance = 1.0e-6f32;

    let full_rank = device.upload(&full_rank_host).unwrap();
    let deficient = device.upload(&deficient_host).unwrap();
    let zero = device.upload(&zero_host).unwrap();
    let full_rank_layout = Layout::c_contiguous([2, 2]).unwrap();
    let deficient_layout = Layout::c_contiguous([3, 3]).unwrap();
    let zero_layout = Layout::c_contiguous([2, 3]).unwrap();

    let leto_full_rank = leto::Array::from_shape_vec([2, 2], full_rank_host).unwrap();
    let leto_deficient = leto::Array::from_shape_vec([3, 3], deficient_host).unwrap();
    let leto_zero = leto::Array::from_shape_vec([2, 3], zero_host).unwrap();

    let expected_full_rank =
        leto_ops::matrix_rank_with_tolerance(&leto_full_rank.view(), tolerance).unwrap();
    let expected_deficient =
        leto_ops::matrix_rank_with_tolerance(&leto_deficient.view(), tolerance).unwrap();
    let expected_zero = leto_ops::matrix_rank_with_tolerance(&leto_zero.view(), tolerance).unwrap();

    let got_full_rank = matrix_rank_with_tolerance(
        &device,
        StridedOperand {
            buffer: &full_rank,
            layout: &full_rank_layout,
        },
        tolerance,
    )
    .unwrap();
    let got_deficient = matrix_rank_with_tolerance(
        &device,
        StridedOperand {
            buffer: &deficient,
            layout: &deficient_layout,
        },
        tolerance,
    )
    .unwrap();
    let got_zero = matrix_rank(
        &device,
        StridedOperand {
            buffer: &zero,
            layout: &zero_layout,
        },
    )
    .unwrap();

    assert_eq!(got_full_rank, expected_full_rank);
    assert_eq!(got_deficient, expected_deficient);
    assert_eq!(got_zero, expected_zero);

    let empty = device.alloc_zeroed::<f32>(1).unwrap();
    let empty_layout = Layout::c_contiguous([0, 3]).unwrap();
    let empty_rank = matrix_rank(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    );
    assert!(matches!(
        empty_rank,
        Err(HephaestusError::DispatchFailed { message }) if message.contains("empty matrix")
    ));
}

#[test]
fn linalg_det_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{det, StridedOperand};
    use leto::Layout;

    let nonsingular_host = vec![2.0f32, 1.0, 3.0, 4.0];
    let singular_host = vec![1.0f32, 2.0, 2.0, 4.0];
    let nonsingular = device.upload(&nonsingular_host).unwrap();
    let singular = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let leto_nonsingular = leto::Array::from_shape_vec([2, 2], nonsingular_host).unwrap();
    let leto_singular = leto::Array::from_shape_vec([2, 2], singular_host).unwrap();
    let expected_nonsingular = leto_ops::det(&leto_nonsingular.view()).unwrap();
    let expected_singular = leto_ops::det(&leto_singular.view()).unwrap();

    let nonsingular_det = det(
        &device,
        StridedOperand {
            buffer: &nonsingular,
            layout: &layout,
        },
    )
    .unwrap();
    let singular_det = det(
        &device,
        StridedOperand {
            buffer: &singular,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got_nonsingular = [0.0f32; 1];
    let mut got_singular = [0.0f32; 1];
    device
        .download(&nonsingular_det, &mut got_nonsingular)
        .unwrap();
    device.download(&singular_det, &mut got_singular).unwrap();
    assert_eq!(got_nonsingular[0], expected_nonsingular);
    assert_eq!(got_singular[0], expected_singular);

    let rectangular = device.alloc_zeroed::<f32>(6).unwrap();
    let rectangular_layout = Layout::c_contiguous([2, 3]).unwrap();
    let rectangular_det = det(
        &device,
        StridedOperand {
            buffer: &rectangular,
            layout: &rectangular_layout,
        },
    );
    assert!(matches!(
        rectangular_det,
        Err(HephaestusError::DispatchFailed { message }) if message.contains("square matrix")
    ));
}

/// Pins the GPU `matrix_rank` relative-threshold contract at the boundary that
/// the residual register flagged as the ill-conditioned divergence risk: a pivot
/// counts toward the rank iff its magnitude exceeds
/// `relative_tolerance * max(abs(matrix))`. For `diag(1, 1, δ)` the max element
/// is `1` and the singular values equal the diagonal magnitudes, so the
/// threshold alone decides the rank and the GPU row-reduction result must agree
/// with Leto's SVD-spectrum result for both tolerances.
#[test]
fn matrix_rank_relative_tolerance_is_the_discriminator() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    // diag(1, 1, 1e-4): max abs = 1, smallest singular value = 1e-4.
    let host = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-4];
    let buffer = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], host).unwrap();

    let tol_keep = 1.0e-6f32; // 1e-4 > 1e-6 * 1 -> small pivot retained
    let tol_drop = 1.0e-2f32; // 1e-4 < 1e-2 * 1 -> small pivot dropped

    let operand = |buffer| StridedOperand {
        buffer,
        layout: &layout,
    };
    let rank_keep = matrix_rank_with_tolerance(&device, operand(&buffer), tol_keep).unwrap();
    let rank_drop = matrix_rank_with_tolerance(&device, operand(&buffer), tol_drop).unwrap();

    assert_eq!(
        rank_keep, 3,
        "1e-4 > 1e-6 * max_abs must count the small pivot"
    );
    assert_eq!(
        rank_drop, 2,
        "1e-4 < 1e-2 * max_abs must drop the small pivot"
    );

    // The diagonal case has singular values == pivot magnitudes, so Leto's
    // SVD-spectrum criterion agrees with the GPU row-reduction threshold.
    let leto_keep = leto_ops::matrix_rank_with_tolerance(&leto_matrix.view(), tol_keep).unwrap();
    let leto_drop = leto_ops::matrix_rank_with_tolerance(&leto_matrix.view(), tol_drop).unwrap();
    assert_eq!(rank_keep, leto_keep);
    assert_eq!(rank_drop, leto_drop);
}

/// Pins the GPU `det` contract on an ill-conditioned (tiny-pivot) matrix: the
/// kernel returns the row-reduction pivot product with no determinant-tolerance
/// zeroing (`det` passes `tolerance == 0`, so only exactly-zero pivots drop), so
/// a near-singular matrix yields its small, nonzero determinant. The input is
/// upper-triangular, so elimination performs no row operations and the result is
/// the analytical pivot product `2 * 3 * δ`.
#[test]
fn det_of_near_singular_triangular_is_exact_pivot_product() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{det, StridedOperand};
    use leto::Layout;

    let delta = 1.0e-5f32;
    let host = vec![2.0f32, 1.0, 5.0, 0.0, 3.0, 7.0, 0.0, 0.0, delta];
    let buffer = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], host).unwrap();

    let det_buf = det(
        &device,
        StridedOperand {
            buffer: &buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got = [0.0f32; 1];
    device.download(&det_buf, &mut got).unwrap();

    // Triangular elimination performs no row operations, so the only error is
    // the f32 rounding of `delta` and the pivot product; a few ulps bound it.
    let analytical = 6.0f32 * delta;
    let tol = analytical.abs() * 8.0 * f32::EPSILON;
    assert!(
        (got[0] - analytical).abs() <= tol,
        "GPU det {} must match analytical pivot product {analytical} within {tol}",
        got[0]
    );
    // The determinant must NOT be zeroed despite being tiny (no det tolerance).
    assert!(
        got[0] > 0.0,
        "near-singular det must stay strictly positive, got {}",
        got[0]
    );
    // Differential: Leto's determinant agrees on this triangular input.
    let leto_det = leto_ops::det(&leto_matrix.view()).unwrap();
    assert!((got[0] - leto_det).abs() <= tol);
}

#[test]
fn cholesky_decomposition_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![6.0f32, 5.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_cholesky = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_cholesky = cholesky_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_cholesky.n(), leto_cholesky.dim());
    assert_eq!(gpu_cholesky.det(), leto_cholesky.det());

    let mut got_lower = vec![0.0f32; 4];
    device
        .download(gpu_cholesky.lower(), &mut got_lower)
        .unwrap();
    let expected_lower = leto::Storage::as_slice(leto_cholesky.lower().storage());
    assert_eq!(got_lower, expected_lower);

    let solution = gpu_cholesky.solve(&device, &rhs).unwrap();
    let expected_solution = leto_cholesky.solve(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; 2];
    device.download(&solution, &mut got_solution).unwrap();
    assert_eq!(
        got_solution,
        leto::Storage::as_slice(expected_solution.storage())
    );

    let inverse = gpu_cholesky.inv(&device).unwrap();
    let expected_inverse = leto_cholesky.inv().unwrap();
    let mut got_inverse = vec![0.0f32; 4];
    device.download(&inverse, &mut got_inverse).unwrap();
    assert_eq!(
        got_inverse,
        leto::Storage::as_slice(expected_inverse.storage())
    );
}

#[test]
fn blocked_cholesky_matches_leto_reference_across_block_boundary() {
    eprintln!("TEST_CHOL: Start");
    let Some(device) = device_or_skip() else {
        eprintln!("TEST_CHOL: Skipped");
        return;
    };
    eprintln!("TEST_CHOL: Device ok");
    use hephaestus_wgpu::{cholesky_decompose_blocked, StridedOperand};
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
    eprintln!("TEST_CHOL: Matrix prepared");
    let matrix = device.upload(&matrix_host).unwrap();
    eprintln!("TEST_CHOL: Matrix uploaded");
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host).unwrap();
    eprintln!("TEST_CHOL: Leto matrix created");
    let leto_cholesky = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();
    eprintln!("TEST_CHOL: Leto cholesky completed");

    let gpu_cholesky = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    eprintln!("TEST_CHOL: GPU cholesky completed");

    let mut got_lower = vec![0.0f32; n * n];
    device
        .download(gpu_cholesky.lower(), &mut got_lower)
        .unwrap();
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

#[test]
fn lu_decomposition_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![2.0f32, 1.0, 4.0, 3.0];
    let rhs_host = vec![3.0f32, 7.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.n(), leto_lu.dim());
    assert_eq!(gpu_lu.det(), leto_lu.det());

    let mut got_factors = vec![0.0f32; 4];
    device.download(gpu_lu.factors(), &mut got_factors).unwrap();
    let expected_factors = leto::Storage::as_slice(leto_lu.factors().storage());
    assert_eq!(got_factors, expected_factors);

    let solution = gpu_lu.solve(&device, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; 2];
    device.download(&solution, &mut got_solution).unwrap();
    assert_eq!(
        got_solution,
        leto::Storage::as_slice(expected_solution.storage())
    );

    let inverse = gpu_lu.inv(&device).unwrap();
    let expected_inverse = leto_lu.inv().unwrap();
    let mut got_inverse = vec![0.0f32; 4];
    device.download(&inverse, &mut got_inverse).unwrap();
    assert_eq!(
        got_inverse,
        leto::Storage::as_slice(expected_inverse.storage())
    );
}

#[test]
fn qr_decomposition_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{qr_decompose, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([3], rhs_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), leto_qr.shape());

    let mut got_r = vec![0.0f32; 6];
    device.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
    let expected_r = leto_qr.r();
    assert_eq!(got_r, leto::Storage::as_slice(expected_r.storage()));

    let solution = gpu_qr.solve_least_squares(&device, &rhs).unwrap();
    let expected_solution = leto_qr.solve_least_squares(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; 2];
    device.download(&solution, &mut got_solution).unwrap();
    assert_eq!(
        got_solution,
        leto::Storage::as_slice(expected_solution.storage())
    );

    let underdetermined = device.alloc_zeroed::<f32>(6).unwrap();
    let underdetermined_layout = Layout::c_contiguous([2, 3]).unwrap();
    let underdetermined_qr = qr_decompose(
        &device,
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

#[test]
fn symmetric_eigen_jacobi_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![
        4.0f32, 1.0, 0.5, 0.25, //
        1.0, 3.0, 0.25, 0.125, //
        0.5, 0.25, 2.0, 0.0625, //
        0.25, 0.125, 0.0625, 1.5,
    ];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([4, 4]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([4, 4], matrix_host).unwrap();
    let leto_eigen = leto_ops::symmetric_eigen_jacobi(&leto_matrix.view()).unwrap();

    let gpu_eigen = symmetric_eigen_jacobi(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_eigen.n(), 4);
    assert_eq!(gpu_eigen.inner().eigenvalues, leto_eigen.eigenvalues);

    let mut got_values = vec![0.0f32; 4];
    device
        .download(gpu_eigen.eigenvalues(), &mut got_values)
        .unwrap();
    assert_eq!(got_values, leto_eigen.eigenvalues);

    let mut got_vectors = vec![0.0f32; 16];
    device
        .download(gpu_eigen.eigenvectors(), &mut got_vectors)
        .unwrap();
    assert_eq!(
        got_vectors,
        leto::Storage::as_slice(leto_eigen.eigenvectors.storage())
    );

    let values_only = symmetric_eigenvalues_jacobi(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let leto_values_only = leto_ops::symmetric_eigenvalues_jacobi(&leto_matrix.view()).unwrap();
    let mut got_values_only = vec![0.0f32; 4];
    device.download(&values_only, &mut got_values_only).unwrap();
    assert_eq!(got_values_only, leto_values_only);
}

#[test]
fn symmetric_eigen_jacobi_rejects_non_symmetric_input() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{symmetric_eigen_jacobi, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 0.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = symmetric_eigen_jacobi(
        &device,
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

#[test]
fn eigenvalues_match_closed_form_diagonal() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{eigenvalues, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![2.0f32, 0.0, 0.0, 3.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let eigen = eigenvalues(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![num_complex::Complex::new(0.0f32, 0.0); 2];
    device.download(&eigen, &mut got).unwrap();
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

#[test]
fn eigenvalues_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{eigenvalues, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![0.0f32, 1.0, -2.0, 3.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let eigen = eigenvalues(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![num_complex::Complex::new(0.0f32, 0.0); 2];
    device.download(&eigen, &mut got).unwrap();

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

#[test]
fn eigenvalues_match_exact_complex_pair_blocks() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{eigenvalues, StridedOperand};
    use leto::Layout;
    use num_complex::Complex;

    let cases: [(usize, Vec<f32>, Vec<Complex<f32>>); 2] = [
        (
            2,
            vec![1.0, -1.0, 1.0, 1.0],
            vec![Complex::new(1.0, -1.0), Complex::new(1.0, 1.0)],
        ),
        (
            3,
            vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            vec![
                Complex::new(0.0, -1.0),
                Complex::new(0.0, 1.0),
                Complex::new(2.0, 0.0),
            ],
        ),
    ];

    for (n, matrix_host, expected) in cases {
        let matrix = device.upload(&matrix_host).unwrap();
        let layout = Layout::c_contiguous([n, n]).unwrap();
        let eigen = eigenvalues(
            &device,
            StridedOperand {
                buffer: &matrix,
                layout: &layout,
            },
        )
        .unwrap();

        let mut got = vec![Complex::new(0.0f32, 0.0); n];
        device.download(&eigen, &mut got).unwrap();
        assert_complex_spectra_close(&got, &expected, 1.0e-5, 1.0e-5);
    }
}

#[test]
fn eigenvalues_match_structured_and_dense_nalgebra_oracles() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{eigenvalues, StridedOperand};
    use leto::Layout;
    use nalgebra::DMatrix;
    use num_complex::Complex;

    let cases: [(usize, Vec<f32>, f32); 4] = [
        (3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0], 1.0e-5),
        (3, vec![2.0, 1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 3.0], 1.0e-5),
        (
            4,
            vec![
                0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 2.0, 1.0,
            ],
            1.0e-5,
        ),
        (
            5,
            (0..25)
                .map(|index| ((index * 7 + 3) % 11) as f32 - 5.0)
                .collect(),
            1.0e-3,
        ),
    ];

    for (n, matrix_host, abs_tol) in cases {
        let matrix = device.upload(&matrix_host).unwrap();
        let layout = Layout::c_contiguous([n, n]).unwrap();
        let eigen = eigenvalues(
            &device,
            StridedOperand {
                buffer: &matrix,
                layout: &layout,
            },
        )
        .unwrap();

        let mut got = vec![Complex::new(0.0f32, 0.0); n];
        device.download(&eigen, &mut got).unwrap();
        let expected: Vec<Complex<f32>> = DMatrix::from_row_slice(n, n, &matrix_host)
            .complex_eigenvalues()
            .iter()
            .copied()
            .collect();
        assert_complex_spectra_close(&got, &expected, abs_tol, 1.0e-4);
    }
}

#[test]
fn eigenvalues_symmetric_input_is_real_and_matches_nalgebra() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{eigenvalues, StridedOperand};
    use leto::Layout;
    use nalgebra::DMatrix;
    use num_complex::Complex;

    let n = 3usize;
    let matrix_host = vec![6.0f32, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let eigen = eigenvalues(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![Complex::new(0.0f32, 0.0); n];
    device.download(&eigen, &mut got).unwrap();
    for value in &got {
        assert!(
            value.im.abs() <= 1.0e-5,
            "symmetric input produced complex eigenvalue {value:?}"
        );
    }
    let expected: Vec<Complex<f32>> = DMatrix::from_row_slice(n, n, &matrix_host)
        .complex_eigenvalues()
        .iter()
        .copied()
        .collect();
    assert_complex_spectra_close(&got, &expected, 1.0e-5, 1.0e-5);
}

#[test]
fn eigenvalues_rejects_non_square_input() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{eigenvalues, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = eigenvalues(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));
}

#[test]
fn singular_values_match_closed_form_diagonal() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{singular_values, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![3.0f32, 0.0, 0.0, 0.0, 2.0, 0.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let values = singular_values(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 2];
    device.download(&values, &mut got).unwrap();
    assert_eq!(got.len(), 2);
    assert_close(got[0], 3.0, 1.0e-5);
    assert_close(got[1], 2.0, 1.0e-5);
}

#[test]
fn svd_decompose_reconstructs_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{svd_decompose, StridedOperand};
    use leto::Layout;

    let rows = 4usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_decompose(
        &device,
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
    device
        .download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();
    device.download(gpu_svd.u(), &mut got_u).unwrap();
    device.download(gpu_svd.v(), &mut got_v).unwrap();

    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-5);
    }

    let reconstructed = reconstruct_svd(&got_u, &got_singular, &got_v, rows, cols);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[test]
fn svd_rank_revealing_accepts_rank_deficient_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{svd_rank_revealing, StridedOperand};
    use leto::Layout;

    let rows = 3usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_svd = svd_rank_revealing(
        &device,
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
    device
        .download(gpu_svd.singular_values(), &mut got_singular)
        .unwrap();

    assert_eq!(rank, 2);
    assert!(got_singular[0] >= got_singular[1]);
    assert_close(got_singular[1], 0.0, 1.0e-5);
    for (actual, expected) in got_singular.iter().zip(leto_svd.singular_values.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[test]
fn bidiagonalize_reconstructs_and_preserves_singular_values() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{bidiagonalize, singular_values, StridedOperand};
    use leto::Layout;

    let rows = 4usize;
    let cols = 3usize;
    let matrix_host = vec![
        4.0f32, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
    ];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();
    let gpu_bd = bidiagonalize(
        &device,
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
    device.download(gpu_bd.u_buffer(), &mut u).unwrap();
    device.download(gpu_bd.b_buffer(), &mut b).unwrap();
    device.download(gpu_bd.v_buffer(), &mut v).unwrap();

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

    let b_buffer = device.upload(&b).unwrap();
    let b_layout = Layout::c_contiguous([rows, cols]).unwrap();
    let sv_b = singular_values(
        &device,
        StridedOperand {
            buffer: &b_buffer,
            layout: &b_layout,
        },
    )
    .unwrap();
    let sv_a = singular_values(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_b = vec![0.0f32; cols];
    let mut got_a = vec![0.0f32; cols];
    device.download(&sv_b, &mut got_b).unwrap();
    device.download(&sv_a, &mut got_a).unwrap();
    for (actual, expected) in got_b.iter().zip(got_a.iter()) {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[test]
fn bidiagonalize_rejects_wide_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{bidiagonalize, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = bidiagonalize(
        &device,
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

#[test]
fn schur_reconstructs_quasi_triangular_and_preserves_spectrum() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{eigenvalues, schur, StridedOperand};
    use leto::Layout;

    let n = 3usize;
    let matrix_host = vec![
        1.0f32, -3.0, 0.0, //
        2.0, 1.0, 0.0, //
        0.0, 0.0, 5.0,
    ];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_schur = schur(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_schur.n(), n);
    let mut q = vec![0.0f32; n * n];
    let mut t = vec![0.0f32; n * n];
    device.download(gpu_schur.q_buffer(), &mut q).unwrap();
    device.download(gpu_schur.t_buffer(), &mut t).unwrap();

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

    let t_buffer = device.upload(&t).unwrap();
    let t_values = eigenvalues(
        &device,
        StridedOperand {
            buffer: &t_buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let a_values = eigenvalues(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_t = vec![num_complex::Complex::new(0.0f32, 0.0); n];
    let mut got_a = vec![num_complex::Complex::new(0.0f32, 0.0); n];
    device.download(&t_values, &mut got_t).unwrap();
    device.download(&a_values, &mut got_a).unwrap();
    assert_complex_spectrum_close(&got_t, &got_a, 1.0e-4);
    device.device().poll(wgpu::PollType::Wait).unwrap();
}

#[test]
fn schur_rejects_rectangular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{schur, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = schur(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );

    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));
    device.device().poll(wgpu::PollType::Wait).unwrap();
}

#[test]
fn hessenberg_reconstructs_and_preserves_similarity_invariants() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{hessenberg, norm_l2, trace, StridedOperand};
    use leto::Layout;

    let n = 4usize;
    let matrix_host = vec![
        4.0f32, 5.0, -2.0, 2.0, //
        1.0, 2.0, 0.0, 1.0, //
        -2.0, 0.0, 3.0, -2.0, //
        2.0, 1.0, -2.0, -1.0,
    ];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_hessenberg = hessenberg(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_hessenberg.n(), n);
    let mut q = vec![0.0f32; n * n];
    let mut h = vec![0.0f32; n * n];
    device.download(gpu_hessenberg.q_buffer(), &mut q).unwrap();
    device.download(gpu_hessenberg.h_buffer(), &mut h).unwrap();

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

    let h_buffer = device.upload(&h).unwrap();
    let h_trace = trace(
        &device,
        StridedOperand {
            buffer: &h_buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let a_trace = trace(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_h_trace = vec![0.0f32; 1];
    let mut got_a_trace = vec![0.0f32; 1];
    device.download(&h_trace, &mut got_h_trace).unwrap();
    device.download(&a_trace, &mut got_a_trace).unwrap();
    assert_close(got_h_trace[0], got_a_trace[0], 1.0e-4);

    let h_norm = norm_l2(
        &device,
        StridedOperand {
            buffer: &h_buffer,
            layout: &layout,
        },
    )
    .unwrap();
    let a_norm = norm_l2(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_h_norm = vec![0.0f32; 1];
    let mut got_a_norm = vec![0.0f32; 1];
    device.download(&h_norm, &mut got_h_norm).unwrap();
    device.download(&a_norm, &mut got_a_norm).unwrap();
    assert_close(got_h_norm[0], got_a_norm[0], 1.0e-3);
}

#[test]
fn hessenberg_rejects_rectangular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{hessenberg, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = hessenberg(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );

    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));
}

#[test]
fn full_piv_lu_reconstructs_and_matches_leto_oracles() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{full_piv_lu, StridedOperand};
    use leto::Layout;

    let n = 4usize;
    let matrix_host = vec![
        2.0f32, 5.0, -2.0, 2.0, //
        1.0, 2.0, 3.0, 1.0, //
        -2.0, 4.0, 3.0, -2.0, //
        2.0, 1.0, -1.0, -1.0,
    ];
    let rhs_host = vec![3.0f32, -1.0, 2.0, 5.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_lu = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.n(), n);
    assert_eq!(gpu_lu.rank(), n);
    let mut packed = vec![0.0f32; n * n];
    device.download(gpu_lu.lu_buffer(), &mut packed).unwrap();
    let lu = packed_lu_product(&packed, n);
    for row in 0..n {
        for col in 0..n {
            let expected =
                matrix_host[gpu_lu.row_permutation()[row] * n + gpu_lu.col_permutation()[col]];
            assert_close(lu[row * n + col], expected, 1.0e-4);
        }
    }

    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([n], rhs_host).unwrap();
    let leto_lu = leto_ops::full_piv_lu(&leto_matrix.view()).unwrap();
    assert_close(gpu_lu.det(), leto_lu.det(), 1.0e-4);

    let solution = gpu_lu.solve(&device, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got_solution = vec![0.0f32; n];
    device.download(&solution, &mut got_solution).unwrap();
    for (actual, expected) in got_solution
        .iter()
        .zip(leto::Storage::as_slice(expected_solution.storage()))
    {
        assert_close(*actual, *expected, 1.0e-4);
    }

    let inverse = gpu_lu.inv(&device).unwrap();
    let expected_inverse = leto_lu.inv().unwrap();
    let mut got_inverse = vec![0.0f32; n * n];
    device.download(&inverse, &mut got_inverse).unwrap();
    for (actual, expected) in got_inverse
        .iter()
        .zip(leto::Storage::as_slice(expected_inverse.storage()))
    {
        assert_close(*actual, *expected, 1.0e-4);
    }
}

#[test]
fn full_piv_lu_reveals_rank_deficiency_and_rejects_inverse() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{full_piv_lu, StridedOperand};
    use leto::Layout;

    let n = 3usize;
    let matrix_host = vec![1.0f32, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let gpu_lu = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_lu.rank(), 2);
    assert_close(gpu_lu.det(), 0.0, 1.0e-5);
    assert!(matches!(
        gpu_lu.inv(&device),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("FullPivLU inverse failed")
    ));
}

#[test]
fn full_piv_lu_rejects_rectangular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{full_piv_lu, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = full_piv_lu(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );

    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));
}

// ── write_buffer tests ────────────────────────────────────────────────

#[test]
fn write_buffer_overwrites_existing_data() {
    let Some(device) = device_or_skip() else {
        return;
    };

    // Upload initial data.
    let initial = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = device.upload(&initial).unwrap();

    // Overwrite with new data via write_buffer.
    let updated = vec![10.0f32, 20.0, 30.0, 40.0];
    device.write_buffer(&buf, &updated).unwrap();

    // Download and verify the overwritten data.
    let mut got = vec![0.0f32; 4];
    device.download(&buf, &mut got).unwrap();
    assert_eq!(got, updated);
}

#[test]
fn write_buffer_rejects_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let wrong_len = vec![1.0f32, 2.0]; // len 2, buffer len 3
    let result: hephaestus_wgpu::Result<()> = device.write_buffer(&buf, &wrong_len);
    assert_length_mismatch(result, 2, 3);
}

#[test]
fn write_buffer_empty_is_noop() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload::<f32>(&[]).unwrap();
    device.write_buffer(&buf, &[] as &[f32]).unwrap();
    assert_eq!(buf.len(), 0);
}

#[test]
fn write_buffer_integer_types() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload(&[0i32, 0, 0]).unwrap();
    let data = vec![42i32, -7, 100];
    device.write_buffer(&buf, &data).unwrap();

    let mut got = vec![0i32; 3];
    device.download(&buf, &mut got).unwrap();
    assert_eq!(got, data);
}

// ── Extended differential decomposition tests ─────────────────────────────

#[test]
fn cholesky_identity_matrix_yields_identity_lower() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_chol = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_chol = cholesky_decompose(
        &device,
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
    device.download(gpu_chol.lower(), &mut got_lower).unwrap();
    let expected_lower = leto::Storage::as_slice(leto_chol.lower().storage());
    assert_eq!(got_lower, expected_lower);
    assert_eq!(got_lower, vec![1.0f32, 0.0, 0.0, 1.0]);
}

#[test]
fn cholesky_spd_reconstruction_matches_original() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose, StridedOperand};
    use leto::Layout;

    // SPD matrix: A = [[4, 2, 0.5], [2, 5, 1], [0.5, 1, 3]]
    let matrix_host = vec![4.0f32, 2.0, 0.5, 2.0, 5.0, 1.0, 0.5, 1.0, 3.0];
    let n = 3;
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host.clone()).unwrap();
    let leto_chol = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_chol = cholesky_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    // Verify L matches leto-ops.
    let mut got_lower = vec![0.0f32; n * n];
    device.download(gpu_chol.lower(), &mut got_lower).unwrap();
    let expected_lower = leto::Storage::as_slice(leto_chol.lower().storage());
    assert_eq!(got_lower, expected_lower);

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

#[test]
fn cholesky_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose, StridedOperand};
    use leto::Layout;

    // A = [[4, 2], [2, 3]], b = [8, 7]  =>  x = [1.25, 1.5].
    // Derivation: eliminating 2x from the second equation gives 2y = 3.
    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![8.0f32, 7.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let gpu_chol = cholesky_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_chol.solve(&device, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();
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

#[test]
fn cholesky_rejects_singular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose, StridedOperand};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose(
        &device,
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

#[test]
fn lu_identity_yields_identity_factors() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose(
        &device,
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
    device.download(gpu_lu.factors(), &mut got_factors).unwrap();
    let expected_factors = leto::Storage::as_slice(leto_lu.factors().storage());
    assert_eq!(got_factors, expected_factors);
}

#[test]
fn lu_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose, StridedOperand};
    use leto::Layout;

    // A = [[2, 1], [4, 3]], b = [5, 11]  =>  x = [2, 1]
    let matrix_host = vec![2.0f32, 1.0, 4.0, 3.0];
    let rhs_host = vec![5.0f32, 11.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_lu.solve(&device, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();
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

#[test]
fn lu_rejects_singular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose, StridedOperand};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = lu_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(result.is_err(), "singular matrix must be rejected by LU");
}

#[test]
fn qr_identity_yields_identity_r() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{qr_decompose, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (2, 2));
    assert_eq!(gpu_qr.shape(), leto_qr.shape());

    let mut got_r = vec![0.0f32; 4];
    device.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
    let r_ref = leto_qr.r();
    let expected_r = leto::Storage::as_slice(r_ref.storage());
    assert_eq!(got_r, expected_r);
}

#[test]
fn qr_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{qr_decompose, StridedOperand};
    use leto::Layout;

    // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]  =>  x = [1, 2]
    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();

    let gpu_qr = qr_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_qr.solve_least_squares(&device, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();

    // Verify A*x ≈ b (residual check).
    let residual_0 = 1.0 * got[0] + 0.0 * got[1] - 1.0;
    let residual_1 = 0.0 * got[0] + 1.0 * got[1] - 2.0;
    let residual_2 = 1.0 * got[0] + 1.0 * got[1] - 3.0;
    assert!(residual_0.abs() <= 1e-4, "QR residual[0] = {residual_0}");
    assert!(residual_1.abs() <= 1e-4, "QR residual[1] = {residual_1}");
    assert!(residual_2.abs() <= 1e-4, "QR residual[2] = {residual_2}");
}

#[test]
fn linalg_norms_match_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{norm_l1, norm_l2, norm_max, StridedOperand};
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

#[test]
fn linalg_reductions_accept_strided_views() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{dot, norm_l1, norm_l2, norm_max, StridedOperand};
    use leto::Layout;

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_host = vec![10.0f32, 20.0, 30.0, 40.0];
    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let reversed = Layout::new([4], [-1], 3);
    let contiguous = Layout::c_contiguous([4]).unwrap();
    let reversed_a = StridedOperand {
        buffer: &a,
        layout: &reversed,
    };
    let contiguous_b = StridedOperand {
        buffer: &b,
        layout: &contiguous,
    };

    let dot_buf = dot(&device, reversed_a, contiguous_b).unwrap();
    let mut got_dot = [0.0f32; 1];
    device.download(&dot_buf, &mut got_dot).unwrap();
    assert_eq!(
        got_dot[0],
        4.0 * 10.0 + 3.0 * 20.0 + 2.0 * 30.0 + 1.0 * 40.0
    );

    let l1_buf = norm_l1(&device, reversed_a).unwrap();
    let mut got_l1 = [0.0f32; 1];
    device.download(&l1_buf, &mut got_l1).unwrap();
    assert_eq!(got_l1[0], 10.0);

    let l2_buf = norm_l2(&device, reversed_a).unwrap();
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

    let max_buf = norm_max(&device, reversed_a).unwrap();
    let mut got_max = [0.0f32; 1];
    device.download(&max_buf, &mut got_max).unwrap();
    assert_eq!(got_max[0], 4.0);
}

// ── Blocked decomposition differential tests ────────────────────────────

#[test]
fn blocked_lu_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose_blocked, StridedOperand};
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
    // Force a pivot swap at the start (row 0) and across the block boundary (row 64)
    matrix_host[0] = 0.0;
    matrix_host[64 * n + 64] = 0.0;

    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();

    let gpu_lu = lu_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    // Solve via host-side decomposition must match.
    let rhs_host = vec![1.0f32; n];
    let rhs = device.upload(&rhs_host).unwrap();
    let solution = gpu_lu.solve(&device, &rhs).unwrap();
    let mut got = vec![0.0f32; n];
    device.download(&solution, &mut got).unwrap();

    // Compute residual: ||A * x - b||_inf
    let mut max_res = 0.0f32;
    for i in 0..n {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += matrix_host[i * n + j] as f64 * got[j] as f64;
        }
        let res = (sum - rhs_host[i] as f64).abs() as f32;
        if res > max_res {
            max_res = res;
        }
    }
    println!("DB: Blocked LU residual norm = {}", max_res);
    assert!(
        max_res <= 1e-3,
        "blocked LU solve is inaccurate, residual norm = {}",
        max_res
    );
}

#[test]
fn blocked_lu_identity_yields_identity_factors() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose_blocked, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose_blocked(
        &device,
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

#[test]
fn blocked_lu_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose_blocked, StridedOperand};
    use leto::Layout;

    // A = [[2, 1], [4, 3]], b = [5, 11]  =>  x = [2, 1]
    let matrix_host = vec![2.0f32, 1.0, 4.0, 3.0];
    let rhs_host = vec![5.0f32, 11.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let leto_lu = leto_ops::lu_decompose(&leto_matrix.view()).unwrap();

    let gpu_lu = lu_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_lu.solve(&device, &rhs).unwrap();
    let expected_solution = leto_lu.solve(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();
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

#[test]
fn blocked_lu_rejects_singular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{lu_decompose_blocked, StridedOperand};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = lu_decompose_blocked(
        &device,
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

#[test]
fn blocked_qr_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{qr_decompose_blocked, StridedOperand};
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
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([m, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([m, n], matrix_host.clone()).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (m, n));

    // R should be upper triangular — check lower triangle is zero.
    let mut got_r = vec![0.0f32; m * n];
    device.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
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
    let rhs = device.upload(&rhs_host).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([m], rhs_host).unwrap();
    let solution = gpu_qr.solve_least_squares(&device, &rhs).unwrap();
    let expected_solution = leto_qr.solve_least_squares(&leto_rhs.view()).unwrap();
    let mut got = vec![0.0f32; n];
    device.download(&solution, &mut got).unwrap();
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

#[test]
fn blocked_qr_identity_yields_identity_r() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{qr_decompose_blocked, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_qr = leto_ops::qr_decompose(&leto_matrix.view()).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    assert_eq!(gpu_qr.shape(), (2, 2));

    let mut got_r = vec![0.0f32; 4];
    device.download(gpu_qr.r_buffer(), &mut got_r).unwrap();
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

#[test]
fn blocked_qr_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{qr_decompose_blocked, StridedOperand};
    use leto::Layout;

    // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 3]  =>  x = [1, 2]
    let matrix_host = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let rhs_host = vec![1.0f32, 2.0, 3.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([3, 2]).unwrap();

    let gpu_qr = qr_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_qr.solve_least_squares(&device, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();

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

#[test]
fn blocked_qr_rejects_underdetermined() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{qr_decompose_blocked, StridedOperand};
    use leto::Layout;

    let host = vec![0.0f32; 6];
    let input = device.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = qr_decompose_blocked(
        &device,
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

#[test]
fn blocked_cholesky_identity_yields_identity_lower() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    let identity_host = vec![1.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&identity_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], identity_host).unwrap();
    let leto_chol = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &device,
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
    device.download(gpu_chol.lower(), &mut got_lower).unwrap();
    assert_eq!(got_lower, vec![1.0f32, 0.0, 0.0, 1.0]);
}

#[test]
fn blocked_cholesky_spd_reconstruction_matches_original() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose_blocked, StridedOperand};
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
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host.clone()).unwrap();
    let leto_chol = leto_ops::cholesky_decompose(&leto_matrix.view()).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    // Reconstruct A' = L * L^T and verify against original.
    let mut got_lower = vec![0.0f32; n * n];
    device.download(gpu_chol.lower(), &mut got_lower).unwrap();
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

#[test]
fn blocked_cholesky_solve_known_system_accurate() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    // A = [[4, 2], [2, 3]], b = [8, 7]  =>  x = [1.25, 1.5]
    let matrix_host = vec![4.0f32, 2.0, 2.0, 3.0];
    let rhs_host = vec![8.0f32, 7.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let rhs = device.upload(&rhs_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let gpu_chol = cholesky_decompose_blocked(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let solution = gpu_chol.solve(&device, &rhs).unwrap();
    let mut got = vec![0.0f32; 2];
    device.download(&solution, &mut got).unwrap();
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

#[test]
fn blocked_cholesky_rejects_singular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{cholesky_decompose_blocked, StridedOperand};
    use leto::Layout;

    let singular_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&singular_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = cholesky_decompose_blocked(
        &device,
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

#[test]
fn col_piv_qr_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{col_piv_qr, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([3, 3], matrix_host).unwrap();
    let leto_decomp = leto_ops::col_piv_qr(&leto_matrix.view()).unwrap();

    let gpu_decomp = col_piv_qr(
        &device,
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
    device.download(gpu_decomp.q(), &mut q).unwrap();
    device.download(gpu_decomp.r(), &mut r).unwrap();

    let q_view = leto_decomp.q();
    let r_view = leto_decomp.r();
    let expected_q = leto::Storage::as_slice(q_view.storage());
    let expected_r = leto::Storage::as_slice(r_view.storage());

    assert_close_slice(&q, expected_q, 1e-4, 0.0);
    assert_close_slice(&r, expected_r, 1e-4, 0.0);
}

#[test]
fn full_piv_lu_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{full_piv_lu, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([2, 2], matrix_host).unwrap();
    let leto_decomp = leto_ops::full_piv_lu(&leto_matrix.view()).unwrap();

    let gpu_decomp = full_piv_lu(
        &device,
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
    device.download(gpu_decomp.lu_buffer(), &mut lu).unwrap();
    assert_close_slice(&lu, leto_decomp.lu_factors(), 1e-4, 0.0);

    let rhs_host = vec![5.0f32, 11.0];
    let rhs = device.upload(&rhs_host).unwrap();
    let sol = gpu_decomp.solve(&device, &rhs).unwrap();
    let mut got_sol = vec![0.0f32; 2];
    device.download(&sol, &mut got_sol).unwrap();

    let leto_rhs = leto::Array::from_shape_vec([2], rhs_host).unwrap();
    let expected_sol = leto_decomp.solve(&leto_rhs.view()).unwrap();
    assert_close_slice(
        &got_sol,
        leto::Storage::as_slice(expected_sol.storage()),
        1e-4,
        0.0,
    );

    let inv = gpu_decomp.inv(&device).unwrap();
    let mut got_inv = vec![0.0f32; 4];
    device.download(&inv, &mut got_inv).unwrap();
    let expected_inv = leto_decomp.inv().unwrap();
    assert_close_slice(
        &got_inv,
        leto::Storage::as_slice(expected_inv.storage()),
        1e-4,
        0.0,
    );
}

#[test]
fn udu_decompose_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{udu_decompose, StridedOperand};
    use leto::Layout;

    let n = 3usize;
    let matrix_host = vec![4.0f32, 2.0, -2.0, 2.0, -3.0, 1.0, -2.0, 1.0, 2.0];
    let rhs_host = vec![3.0f32, -1.0, 2.0];
    let matrix = device.upload(&matrix_host).unwrap();
    device.device().poll(wgpu::PollType::Wait).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host.clone()).unwrap();
    let leto_decomp = leto_ops::udu_decompose(&leto_matrix.view()).unwrap();

    let gpu_decomp = udu_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    device.device().poll(wgpu::PollType::Wait).unwrap();

    assert_eq!(gpu_decomp.n(), n);
    assert_close(gpu_decomp.det(), leto_decomp.det(), 1.0e-4);

    let mut u = vec![0.0f32; n * n];
    let mut d = vec![0.0f32; n];
    device.download(gpu_decomp.u_buffer(), &mut u).unwrap();
    device.download(gpu_decomp.d_buffer(), &mut d).unwrap();

    let u_view = leto_decomp.u();
    let expected_u = leto::Storage::as_slice(u_view.storage());
    assert_close_slice(&u, expected_u, 1e-4, 0.0);
    assert_close_slice(&d, leto_decomp.diagonal(), 1e-4, 0.0);

    let reconstructed = udu_transpose_product(&u, &d, n);
    for (actual, expected) in reconstructed.iter().zip(matrix_host.iter()) {
        assert_close(*actual, *expected, 1.0e-3);
    }

    let rhs = device.upload(&rhs_host).unwrap();
    let x = gpu_decomp.solve(&device, &rhs).unwrap();
    let mut got_x = vec![0.0f32; n];
    device.download(&x, &mut got_x).unwrap();
    let leto_rhs = leto::Array::from_shape_vec([n], rhs_host).unwrap();
    let expected_x = leto_decomp.solve(&leto_rhs.view()).unwrap();
    assert_close_slice(
        &got_x,
        leto::Storage::as_slice(expected_x.storage()),
        1e-4,
        0.0,
    );

    let inv = gpu_decomp.inv(&device).unwrap();
    let mut got_inv = vec![0.0f32; n * n];
    device.download(&inv, &mut got_inv).unwrap();
    let expected_inv = leto_decomp.inv().unwrap();
    assert_close_slice(
        &got_inv,
        leto::Storage::as_slice(expected_inv.storage()),
        1e-4,
        0.0,
    );
}

#[test]
fn udu_decompose_rejects_invalid_contracts() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{udu_decompose, StridedOperand};
    use leto::Layout;

    let rectangular_host = vec![1.0f32, 2.0, 3.0, 2.0, 4.0, 5.0];
    let rectangular = device.upload(&rectangular_host).unwrap();
    let rectangular_layout = Layout::c_contiguous([2, 3]).unwrap();
    let rectangular_result = udu_decompose(
        &device,
        StridedOperand {
            buffer: &rectangular,
            layout: &rectangular_layout,
        },
    );
    assert!(matches!(
        rectangular_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));

    let nonsymmetric_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let nonsymmetric = device.upload(&nonsymmetric_host).unwrap();
    let nonsymmetric_layout = Layout::c_contiguous([2, 2]).unwrap();
    let nonsymmetric_result = udu_decompose(
        &device,
        StridedOperand {
            buffer: &nonsymmetric,
            layout: &nonsymmetric_layout,
        },
    );
    assert!(matches!(
        nonsymmetric_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("UDU decomposition failed")
    ));

    let zero_pivot_host = vec![1.0f32, 1.0, 1.0, 0.0];
    let zero_pivot = device.upload(&zero_pivot_host).unwrap();
    let zero_pivot_layout = Layout::c_contiguous([2, 2]).unwrap();
    let zero_pivot_result = udu_decompose(
        &device,
        StridedOperand {
            buffer: &zero_pivot,
            layout: &zero_pivot_layout,
        },
    );
    assert!(matches!(
        zero_pivot_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("UDU decomposition failed")
    ));
}

#[test]
fn bunch_kaufman_matches_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{bunch_kaufman, StridedOperand};
    use leto::Layout;

    let n = 3usize;
    let matrix_host = vec![0.1f32, 10.0, 0.0, 10.0, 1000.0, 0.0, 0.0, 0.0, 2.0];
    let matrix = device.upload(&matrix_host).unwrap();
    device.device().poll(wgpu::PollType::Wait).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host.clone()).unwrap();
    let leto_decomp = leto_ops::bunch_kaufman(&leto_matrix.view()).unwrap();

    let gpu_decomp = bunch_kaufman(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();
    device.device().poll(wgpu::PollType::Wait).unwrap();

    assert_eq!(gpu_decomp.n(), n);
    assert_eq!(gpu_decomp.permutation(), leto_decomp.permutation());

    let mut l = vec![0.0f32; n * n];
    let mut d = vec![0.0f32; n * n];
    device.download(gpu_decomp.l_buffer(), &mut l).unwrap();
    device.download(gpu_decomp.d_buffer(), &mut d).unwrap();

    let l_view = leto_decomp.l();
    let d_view = leto_decomp.d();
    let expected_l = leto::Storage::as_slice(l_view.storage());
    let expected_d = leto::Storage::as_slice(d_view.storage());
    assert_close_slice(&l, expected_l, 1e-4, 0.0);
    assert_close_slice(&d, expected_d, 1e-4, 0.0);

    let reconstructed = ldl_transpose_product(&l, &d, n);
    for row in 0..n {
        for col in 0..n {
            let expected =
                matrix_host[gpu_decomp.permutation()[row] * n + gpu_decomp.permutation()[col]];
            assert_close(reconstructed[row * n + col], expected, 1.0e-3);
        }
    }
}

#[test]
fn bunch_kaufman_rejects_rectangular_and_nonsymmetric() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{bunch_kaufman, StridedOperand};
    use leto::Layout;

    let rectangular_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rectangular = device.upload(&rectangular_host).unwrap();
    let rectangular_layout = Layout::c_contiguous([2, 3]).unwrap();
    let rectangular_result = bunch_kaufman(
        &device,
        StridedOperand {
            buffer: &rectangular,
            layout: &rectangular_layout,
        },
    );
    assert!(matches!(
        rectangular_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("square matrix")
    ));

    let nonsymmetric_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let nonsymmetric = device.upload(&nonsymmetric_host).unwrap();
    let nonsymmetric_layout = Layout::c_contiguous([2, 2]).unwrap();
    let nonsymmetric_result = bunch_kaufman(
        &device,
        StridedOperand {
            buffer: &nonsymmetric,
            layout: &nonsymmetric_layout,
        },
    );
    assert!(matches!(
        nonsymmetric_result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Bunch-Kaufman decomposition failed")
    ));
}

#[test]
fn linalg_pinv_matches_closed_form_diagonal() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{pinv, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![2.0f32, 0.0, 0.0, 4.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = pinv(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 4];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![0.5, 0.0, 0.0, 0.25]);
}

#[test]
fn linalg_pinv_rank_deficient_satisfies_moore_penrose() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{pinv, StridedOperand};
    use leto::Layout;
    use nalgebra::DMatrix;

    let n = 2usize;
    let matrix_host = vec![1.0f32, 2.0, 2.0, 4.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();

    let out = pinv(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; n * n];
    device.download(&out, &mut got).unwrap();

    let expected = DMatrix::from_row_slice(n, n, &matrix_host)
        .pseudo_inverse(1.0e-6)
        .unwrap();
    let mut expected_host = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            expected_host.push(expected[(row, col)]);
        }
    }
    assert_close_slice(&got, &expected_host, 1.0e-4, 1.0e-4);

    let a_ap = matmul_host(&matrix_host, n, n, &got, n);
    let a_ap_a = matmul_host(&a_ap, n, n, &matrix_host, n);
    assert_close_slice(&a_ap_a, &matrix_host, 1.0e-4, 1.0e-4);

    let ap_a = matmul_host(&got, n, n, &matrix_host, n);
    let ap_a_ap = matmul_host(&ap_a, n, n, &got, n);
    assert_close_slice(&ap_a_ap, &got, 1.0e-4, 1.0e-4);
}

#[test]
fn linalg_pinv_handles_rectangular_full_rank_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{pinv, StridedOperand};
    use leto::Layout;
    use nalgebra::DMatrix;

    let rows = 3usize;
    let cols = 2usize;
    let matrix_host = vec![1.0f32, 2.0, 0.0, 1.0, 2.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([rows, cols]).unwrap();

    let out = pinv(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; cols * rows];
    device.download(&out, &mut got).unwrap();

    let expected = DMatrix::from_row_slice(rows, cols, &matrix_host)
        .pseudo_inverse(1.0e-6)
        .unwrap();
    let mut expected_host = Vec::with_capacity(cols * rows);
    for row in 0..cols {
        for col in 0..rows {
            expected_host.push(expected[(row, col)]);
        }
    }
    assert_close_slice(&got, &expected_host, 1.0e-4, 1.0e-4);

    let a_ap = matmul_host(&matrix_host, rows, cols, &got, rows);
    let a_ap_a = matmul_host(&a_ap, rows, rows, &matrix_host, cols);
    assert_close_slice(&a_ap_a, &matrix_host, 1.0e-4, 1.0e-4);
}

#[test]
fn linalg_pinv_rejects_non_finite_input() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{pinv, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![1.0f32, f32::NAN, 0.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let result = pinv(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    );
    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message })
            if message.contains("Pseudoinverse failed")
    ));
}

#[test]
fn linalg_matexp_matches_closed_form_diagonal() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{matexp, StridedOperand};
    use leto::Layout;

    let matrix_host = vec![0.0f32, 0.0, 0.0, 1.0];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = matexp(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; 4];
    device.download(&out, &mut got).unwrap();
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
fn linalg_matexp_matches_nilpotent_and_rotation_closed_forms() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{matexp, StridedOperand};
    use leto::Layout;

    let layout = Layout::c_contiguous([2, 2]).unwrap();

    let nilpotent_host = vec![0.0f32, 1.0, 0.0, 0.0];
    let nilpotent = device.upload(&nilpotent_host).unwrap();
    let nilpotent_out = matexp(
        &device,
        StridedOperand {
            buffer: &nilpotent,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_nilpotent = vec![0.0f32; 4];
    device.download(&nilpotent_out, &mut got_nilpotent).unwrap();
    assert_close_slice(&got_nilpotent, &[1.0, 1.0, 0.0, 1.0], 1.0e-5, 0.0);

    let theta = 0.9f32;
    let skew_host = vec![0.0f32, -theta, theta, 0.0];
    let skew = device.upload(&skew_host).unwrap();
    let skew_out = matexp(
        &device,
        StridedOperand {
            buffer: &skew,
            layout: &layout,
        },
    )
    .unwrap();
    let mut got_skew = vec![0.0f32; 4];
    device.download(&skew_out, &mut got_skew).unwrap();
    assert_close_slice(
        &got_skew,
        &[theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
        1.0e-5,
        1.0e-5,
    );
}

#[test]
fn linalg_matexp_matches_nalgebra_general_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{matexp, StridedOperand};
    use leto::Layout;
    use nalgebra::DMatrix;

    let n = 3usize;
    let matrix_host = vec![1.2f32, -0.7, 0.4, 0.3, 2.1, -1.5, -0.6, 0.8, 0.5];
    let matrix = device.upload(&matrix_host).unwrap();
    let layout = Layout::c_contiguous([n, n]).unwrap();

    let out = matexp(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let mut got = vec![0.0f32; n * n];
    device.download(&out, &mut got).unwrap();

    let expected = DMatrix::from_row_slice(n, n, &matrix_host).exp();
    let mut expected_host = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            expected_host.push(expected[(row, col)]);
        }
    }
    assert_close_slice(&got, &expected_host, 1.0e-3, 1.0e-4);
}

#[test]
fn linalg_matexp_rejects_invalid_contracts() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{matexp, StridedOperand};
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

#[test]
fn test_wgpu_uniform_and_normal_with_seed() {
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

#[test]
fn test_wgpu_sparse_matrix_spmv_spmm() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{spmm, spmv, GpuCsrMatrix, StridedOperand};
    use leto::Layout;

    // Create a 3x3 diagonal-ish matrix:
    // [ 2.0  0.0 -1.0 ]
    // [ 0.0  3.0  0.0 ]
    // [ 0.0  0.0  4.0 ]
    let dense_host = vec![2.0f32, 0.0, -1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let cpu_csr = leto_ops::CsrMatrix::from_dense(&leto::ArrayView2::new(layout, &dense_host));

    let gpu_csr = GpuCsrMatrix::from_cpu(&device, &cpu_csr).unwrap();
    assert_eq!(gpu_csr.shape(), (3, 3));
    assert_eq!(gpu_csr.nnz(), 4);

    // Round-trip back to CPU
    let cpu_csr_2 = gpu_csr.to_cpu(&device).unwrap();
    assert_eq!(cpu_csr, cpu_csr_2);

    // SpMV: y = A * x, x = [1.0, 2.0, 3.0]
    // Expected y = [ 2*1 - 3, 3*2, 4*3 ] = [ -1.0, 6.0, 12.0 ]
    let x_host = vec![1.0f32, 2.0, 3.0];
    let x_buf = device.upload(&x_host).unwrap();
    let y_buf = spmv(&device, &gpu_csr, &x_buf).unwrap();
    let mut got_y = vec![0.0f32; 3];
    device.download(&y_buf, &mut got_y).unwrap();
    assert_close_slice(&got_y, &[-1.0, 6.0, 12.0], 1.0e-4, 1.0e-4);

    // SpMM: C = A * B, B = [ 1.0  2.0 ]
    //                      [ 3.0  4.0 ]
    //                      [ 5.0  6.0 ]
    // Expected C = [ 2*1 - 5, 2*2 - 6 ] = [ -3.0, -2.0 ]
    //              [ 3*3,     3*4     ]   [  9.0, 12.0 ]
    //              [ 4*5,     4*6     ]   [ 20.0, 24.0 ]
    let b_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_buf = device.upload(&b_host).unwrap();
    let b_layout = Layout::c_contiguous([3, 2]).unwrap();
    let b_op = StridedOperand {
        buffer: &b_buf,
        layout: &b_layout,
    };
    let c_buf = spmm(&device, &gpu_csr, &b_op).unwrap();
    let mut got_c = vec![0.0f32; 6];
    device.download(&c_buf, &mut got_c).unwrap();
    assert_close_slice(&got_c, &[-3.0, -2.0, 9.0, 12.0, 20.0, 24.0], 1.0e-4, 1.0e-4);
}
