use super::*;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, eunomia::Pod, eunomia::Zeroable)]
pub(crate) struct AlternateIdentity(pub(crate) i32);

/// Acquire a device, or `None` to skip (no `cuda` feature / no GPU).
pub(crate) fn device(test: &str) -> Option<CudaDevice> {
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

pub(crate) fn assert_elementwise_alias_rejected(result: Result<()>) {
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

pub(crate) fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
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

pub(crate) fn assert_dispatch_message<T>(result: Result<T>, expected: &str) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => assert_eq!(message, expected),
        Err(error) => panic!("expected dispatch failure {expected:?}, got {error:?}"),
        Ok(_) => panic!("expected dispatch failure {expected:?}, got success"),
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "got {actual}, expected {expected}, tolerance {tolerance}"
    );
}

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

#[cfg(feature = "decomposition")]
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

/// Shared adversarial-layout driver: every non-dense view (transposed,
/// offset, broadcast/zero-stride) must be rejected by the blocked entry
/// points with the typed dense-operand error BEFORE any device copy. The
/// broadcast case is the memory-safety case: its validated storage extent
/// (4 elements here) is smaller than rows*cols, so the former raw
/// whole-matrix copy would have read past the allocation.
#[cfg(feature = "decomposition")]
pub(crate) fn assert_blocked_rejects_non_dense<F, O>(dev: &CudaDevice, entry: F, label: &str)
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

pub(crate) fn assert_vector_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * want.abs().max(1.0),
            "index {index}: got {got}, expected {want}"
        );
    }
}
