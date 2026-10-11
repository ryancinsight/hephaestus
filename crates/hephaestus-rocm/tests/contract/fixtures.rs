use super::*;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, eunomia::Pod, eunomia::Zeroable)]
pub(crate) struct AlternateIdentity(pub(crate) i32);

pub(crate) struct StreamAddKernel;

pub(crate) struct GroupedStreamAddKernel;

pub(crate) fn device(test: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_some() {
                panic!("ROCm device required for {test}, but acquisition failed: {error}");
            }
            eprintln!("skip {test}: ROCm device unavailable ({error})");
            None
        }
    }
}

pub(crate) fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
    match result {
        Err(HephaestusError::LengthMismatch {
            host_len: actual_host_len,
            device_len: actual_device_len,
        }) => {
            assert_eq!(actual_host_len, host_len);
            assert_eq!(actual_device_len, device_len);
        }
        Err(error) => panic!("expected length mismatch, got {error:?}"),
        Ok(_) => panic!("expected length mismatch {host_len}->{device_len}, got success"),
    }
}

pub(crate) fn assert_near(actual: f32, expected: f32, ulps: f32) {
    let tolerance = ulps * f32::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "got {actual}, expected {expected}, tolerance {tolerance}"
    );
}

#[cfg(feature = "decomposition")]
pub(crate) fn reconstruct_svd(
    u: &[f32],
    v: &[f32],
    singular_values: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let rank = singular_values.len();
    (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                (0..rank)
                    .map(|k| u[row * rank + k] * singular_values[k] * v[col * rank + k])
                    .sum()
            })
        })
        .collect()
}

#[cfg(feature = "decomposition")]
pub(crate) fn matmul_square(lhs: &[f32], rhs: &[f32], n: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|row| {
            (0..n).map(move |col| {
                (0..n)
                    .map(|index| lhs[row * n + index] * rhs[index * n + col])
                    .sum()
            })
        })
        .collect()
}

#[cfg(feature = "decomposition")]
pub(crate) fn transpose_square(matrix: &[f32], n: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|row| (0..n).map(move |col| matrix[col * n + row]))
        .collect()
}

#[cfg(feature = "decomposition")]
pub(crate) fn assert_orthogonal(matrix: &[f32], n: usize) {
    let transpose = transpose_square(matrix, n);
    let product = matmul_square(&transpose, matrix, n);
    for row in 0..n {
        for col in 0..n {
            assert_near(
                product[row * n + col],
                if row == col { 1.0 } else { 0.0 },
                8192.0,
            );
        }
    }
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
    actual.sort_by(|lhs, rhs| {
        lhs.re
            .total_cmp(&rhs.re)
            .then_with(|| lhs.im.total_cmp(&rhs.im))
    });
    expected.sort_by(|lhs, rhs| {
        lhs.re
            .total_cmp(&rhs.re)
            .then_with(|| lhs.im.total_cmp(&rhs.im))
    });
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_near(actual.re, expected.re, tolerance);
        assert_near(actual.im, expected.im, tolerance);
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
