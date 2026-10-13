use super::*;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, eunomia::Pod, eunomia::Zeroable)]
pub(crate) struct AlternateIdentity(pub(crate) i32);

pub(crate) fn device_or_skip() -> Option<WgpuDevice> {
    super::super::device_or_skip()
}

pub(crate) fn assert_elementwise_alias_rejected(result: hephaestus_wgpu::Result<()>) {
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

pub(crate) fn assert_length_mismatch<T>(
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

pub(crate) fn assert_dispatch_message<T>(result: hephaestus_wgpu::Result<T>, expected: &str) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => assert_eq!(message, expected),
        Err(error) => panic!("expected dispatch failure {expected:?}, got {error:?}"),
        Ok(_) => panic!("expected dispatch failure {expected:?}, got success"),
    }
}

pub(crate) fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "got {actual}, expected {expected}, tolerance {tolerance}"
    );
}

#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub(crate) fn assert_close_slice(got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
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
pub(crate) fn assert_complex_spectra_close(
    got: &[eunomia::Complex<f32>],
    expected: &[eunomia::Complex<f32>],
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

#[cfg(feature = "decomposition")]
pub(crate) fn reconstruct_svd(
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

#[cfg(any(feature = "decomposition", feature = "sparse"))]
pub(crate) fn matmul_host(
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
pub(crate) fn transpose_host(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = matrix[row * cols + col];
        }
    }
    out
}

#[cfg(feature = "decomposition")]
pub(crate) fn assert_orthogonal_host(matrix: &[f32], n: usize, tolerance: f32) {
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
pub(crate) fn sort_complex(values: &mut [eunomia::Complex<f32>]) {
    values.sort_by(|lhs, rhs| {
        lhs.re
            .total_cmp(&rhs.re)
            .then_with(|| lhs.im.total_cmp(&rhs.im))
    });
}

#[cfg(feature = "decomposition")]
pub(crate) fn assert_complex_spectrum_close(
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

#[cfg(feature = "decomposition")]
pub(crate) fn packed_lu_product(packed: &[f32], n: usize) -> Vec<f32> {
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

#[cfg(feature = "decomposition")]
pub(crate) fn qr_r_buffer_contract_at_shape(
    device: &hephaestus_wgpu::WgpuDevice,
    m: usize,
    n: usize,
    blocked_route: bool,
) {
    use hephaestus_wgpu::{StridedOperand, qr_decompose, qr_decompose_blocked};
    use leto::Layout;

    assert_eq!(
        n.div_ceil(32) > 4 && m >= 2048,
        blocked_route,
        "fixture [{m}, {n}] must land in the intended routing regime"
    );

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
    let operand = || StridedOperand {
        buffer: &matrix,
        layout: &layout,
    };

    let direct = qr_decompose(device, operand()).unwrap();
    let blocked = qr_decompose_blocked(device, operand()).unwrap();
    assert_eq!(direct.shape(), (m, n));
    assert_eq!(blocked.shape(), (m, n));

    let mut direct_r = vec![0.0f32; m * n];
    let mut blocked_r = vec![0.0f32; m * n];
    device.download(direct.r_buffer(), &mut direct_r).unwrap();
    device.download(blocked.r_buffer(), &mut blocked_r).unwrap();

    for row in 1..m {
        for col in 0..n.min(row) {
            for (label, values) in [("direct", &direct_r), ("blocked", &blocked_r)] {
                let value = values[row * n + col];
                assert!(
                    value.abs() <= f32::EPSILON,
                    "{label} R[{row}, {col}] = {value} must be zero at [{m}, {n}]: r_buffer()                      is the upper-triangular factor on both entry points"
                );
            }
        }
    }

    // The device buffer and the host factor's R are the same object written to
    // two places: each R row is produced once — by leto on the delegating
    // route, by the panel factorisation on the device route — and stored to
    // both `packed` and the buffer. Callers that return `r_buffer()` instead
    // of uploading `inner().r()`, as the Python binding does, depend on that
    // identity, so it is asserted exactly rather than within a tolerance.
    let blocked_inner_r = blocked.inner().r();
    let blocked_inner_values = leto::Storage::as_slice(blocked_inner_r.storage());
    assert_eq!(blocked_inner_r.shape(), [m, n]);
    assert_eq!(
        blocked_r, blocked_inner_values,
        "the device R buffer must equal the host factor's R exactly at [{m}, {n}]"
    );

    // Both factor the same matrix by the same Householder scheme, so their R
    // entries differ only by the order of the arithmetic. Householder QR is
    // columnwise backward stable — ‖ΔR·eⱼ‖₂ ≤ c(m, n)·ε·‖aⱼ‖₂ (Higham,
    // *Accuracy and Stability*, ch. 19) — and this fixture's column norms are
    // bounded by 5.1, so 2·c(m, n)·ε·5.1 with c(m, n) ≤ m·n covers both
    // factorisations' error against the exact R and hence their difference.
    let tolerance = 2.0 * (m * n) as f32 * f32::EPSILON * 5.1;
    for row in 0..n {
        for col in 0..n {
            let index = row * n + col;
            let delta = (direct_r[index] - blocked_r[index]).abs();
            assert!(
                delta <= tolerance,
                "entry points disagree at R[{row}, {col}] for [{m}, {n}]: direct {},                  blocked {}, delta {delta} exceeds {tolerance}",
                direct_r[index],
                blocked_r[index]
            );
        }
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn qr_accumulated_q_contract_at_shape(
    device: &hephaestus_wgpu::WgpuDevice,
    m: usize,
    n: usize,
    blocked_route: bool,
) {
    use hephaestus_wgpu::{StridedOperand, qr_decompose_blocked};
    use leto::Layout;

    assert_eq!(
        n.div_ceil(32) > 4 && m >= 2048,
        blocked_route,
        "fixture [{m}, {n}] must land in the intended routing regime"
    );

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
    let decomp = qr_decompose_blocked(
        device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .unwrap();

    let q_buffer = decomp.accumulate_q(device).unwrap();
    let mut q_gpu = vec![0.0f32; m * m];
    device.download(&q_buffer, &mut q_gpu).unwrap();

    let q_host_array = decomp.inner().q();
    assert_eq!(q_host_array.shape(), [m, m]);
    let q_host = leto::Storage::as_slice(q_host_array.storage());

    // Both apply the same min(m, n) reflectors to the same identity and differ
    // only in how each dot product vᵀ·q is summed — the host sequentially, the
    // device as a 256-way tree. Accumulating a product of computed Householder
    // reflectors is backward stable with ‖Q̂ − Q‖ ≤ c(m, n)·ε, c(m, n) ≤
    // m·min(m, n) (Higham, *Accuracy and Stability*, ch. 19): each reflector is
    // orthogonal, so earlier rounding is transported rather than amplified, and
    // Q's columns are unit-norm, so the ‖v‖₂‖q‖₂ factors are O(1). The tree
    // reduction's own bound is the smaller of the two (depth log₂256 + m/256
    // against the host's m), so the sum of both bounds covers their difference.
    let tolerance = 2.0 * (m * n.min(m)) as f32 * f32::EPSILON;
    for row in 0..m {
        for col in 0..m {
            let index = row * m + col;
            let delta = (q_gpu[index] - q_host[index]).abs();
            assert!(
                delta <= tolerance,
                "device Q disagrees with the host reference at Q[{row}, {col}] for [{m}, {n}]: device {}, host {}, delta {delta} exceeds {tolerance}",
                q_gpu[index],
                q_host[index]
            );
        }
    }

    // Orthogonality (QᵀQ against I, order-independent) lives in the
    // `qr::tests::blocked_schedule_accumulates_orthogonal_q` unit test, which
    // drives the shared schedule directly at 160×129: five panels exercise
    // the multi-panel walk while the m×m Gram stays cheap. Repeating it here
    // would pay a 2048³ Gram for the same schedule property.
}

/// Shared adversarial-layout driver: every non-dense view (transposed,
/// offset, broadcast/zero-stride) must be rejected by the blocked entry
/// points with the typed dense-operand error BEFORE any raw device copy.
/// The broadcast case is the wrong-extent case: its validated storage
/// extent (4 elements here) is smaller than rows*cols, so the former raw
/// whole-matrix copy exceeded the operand's storage.
#[cfg(feature = "decomposition")]
pub(crate) fn assert_blocked_rejects_non_dense<F, O>(device: &WgpuDevice, entry: F, label: &str)
where
    F: Fn(&WgpuDevice, hephaestus_wgpu::StridedOperand<'_, f32, 2>) -> hephaestus_core::Result<O>,
{
    use leto::Layout;

    let dense_host: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
    let dense_buf = device.upload(&dense_host).unwrap();
    let small_host = [1.0f32, 2.0, 3.0, 4.0];
    let small_buf = device.upload(&small_host).unwrap();

    let transposed = Layout::try_new([4, 4], [1, 4], 0).expect("valid test layout");
    let offset = Layout::try_new([3, 3], [4, 1], 5).expect("valid test layout");
    let broadcast = Layout::try_new([4, 4], [0, 1], 0).expect("valid test layout");

    for (name, layout, buffer) in [
        ("transposed", &transposed, &dense_buf),
        ("offset", &offset, &dense_buf),
        ("broadcast", &broadcast, &small_buf),
    ] {
        let result = entry(device, hephaestus_wgpu::StridedOperand { buffer, layout });
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
