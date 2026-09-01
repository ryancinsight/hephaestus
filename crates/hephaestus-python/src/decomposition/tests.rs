use super::*;
use crate::device::PyDevice;
use crate::test_support::prepare_python;

fn cuda_device_or_skip() -> Option<PyDevice> {
    match PyDevice::new(Some("cuda")) {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA Python QR test: device unavailable ({error})");
            None
        }
        Err(error) => panic!("CUDA Python QR test requires a physical device: {error}"),
    }
}

#[test]
fn cuda_qr_binding_returns_device_factors_that_reconstruct_input() {
    prepare_python();
    Python::attach(|py| {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (rows, cols) = (5, 3);
        let input = vec![
            4.0, 1.0, 0.5, 0.25, 3.0, 0.75, 0.5, 0.75, 2.5, 1.0, -0.5, 0.25, 0.75, 0.5, 1.5,
        ];
        let array = PyArray {
            buffer: device.inner.upload_f32(&input).expect("CUDA input upload"),
            device: device.inner.clone(),
            shape: vec![rows, cols],
        };

        let (q, r) = qr(py, &array).expect("Python CUDA QR");
        assert_eq!(q.shape, [rows, rows]);
        assert_eq!(r.shape, [rows, cols]);
        assert!(matches!(&q.buffer, BackendBuffer::Cuda(_)));
        assert!(matches!(&r.buffer, BackendBuffer::Cuda(_)));

        let q_values = q.device.download_f32(&q.buffer).expect("Q value download");
        let r_values = r.device.download_f32(&r.buffer).expect("R value download");
        let max_input = input.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        // Householder QR is backward stable with O(mn*epsilon) elementwise
        // error for this well-scaled fixture. Four times that bound covers
        // factorization, Q accumulation, and the reconstruction dot product.
        let tolerance = 4.0 * (rows * cols) as f32 * f32::EPSILON * max_input;
        for row in 0..rows {
            for column in 0..cols {
                let reconstructed = (0..rows)
                    .map(|inner| q_values[row * rows + inner] * r_values[inner * cols + column])
                    .sum::<f32>();
                let expected = input[row * cols + column];
                let delta = (reconstructed - expected).abs();
                assert!(
                    delta <= tolerance,
                    "QR reconstruction [{row}, {column}] is {reconstructed}, expected {expected}; delta {delta} exceeds {tolerance}"
                );
            }
        }
    });
}
