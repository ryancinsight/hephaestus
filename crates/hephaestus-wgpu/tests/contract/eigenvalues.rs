use super::*;

#[cfg(feature = "decomposition")]
pub(crate) fn symmetric_eigen_jacobi_rejects_non_symmetric_input() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, symmetric_eigen_jacobi};
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

#[cfg(feature = "decomposition")]
pub(crate) fn eigenvalues_match_closed_form_diagonal() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, eigenvalues};
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

    let mut got = vec![eunomia::Complex::new(0.0f32, 0.0); 2];
    device.download(&eigen, &mut got).unwrap();
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
pub(crate) fn eigenvalues_match_exact_complex_pair_blocks() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use eunomia::Complex;
    use hephaestus_wgpu::{StridedOperand, eigenvalues};
    use leto::Layout;

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

#[cfg(feature = "decomposition")]
pub(crate) fn eigenvalues_match_structured_and_dense_leto_oracles() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use eunomia::Complex;
    use hephaestus_wgpu::{StridedOperand, eigenvalues};
    use leto::Layout;

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
        let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host).unwrap();
        let expected: Vec<Complex<f32>> = leto_ops::eigenvalues(&leto_matrix.view())
            .unwrap()
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();
        assert_complex_spectra_close(&got, &expected, abs_tol, 1.0e-4);
    }
}

#[cfg(feature = "decomposition")]
pub(crate) fn eigenvalues_symmetric_input_is_real_and_matches_leto() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use eunomia::Complex;
    use hephaestus_wgpu::{StridedOperand, eigenvalues};
    use leto::Layout;

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
    let leto_matrix = leto::Array::from_shape_vec([n, n], matrix_host).unwrap();
    let expected: Vec<Complex<f32>> = leto_ops::eigenvalues(&leto_matrix.view())
        .unwrap()
        .iter()
        .map(|c| Complex::new(c.re, c.im))
        .collect();
    assert_complex_spectra_close(&got, &expected, 1.0e-5, 1.0e-5);
}

#[cfg(feature = "decomposition")]
pub(crate) fn eigenvalues_rejects_non_square_input() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, eigenvalues};
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

#[cfg(feature = "decomposition")]
pub(crate) fn schur_reconstructs_quasi_triangular_and_preserves_spectrum() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, eigenvalues, schur};
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
    let mut got_t = vec![eunomia::Complex::new(0.0f32, 0.0); n];
    let mut got_a = vec![eunomia::Complex::new(0.0f32, 0.0); n];
    device.download(&t_values, &mut got_t).unwrap();
    device.download(&a_values, &mut got_a).unwrap();
    assert_complex_spectrum_close(&got_t, &got_a, 1.0e-4);
    device
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[cfg(feature = "decomposition")]
pub(crate) fn schur_rejects_rectangular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, schur};
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
    device
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

#[cfg(feature = "decomposition")]
pub(crate) fn hessenberg_reconstructs_and_preserves_similarity_invariants() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, hessenberg, norm_l2, trace};
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

#[cfg(feature = "decomposition")]
pub(crate) fn hessenberg_rejects_rectangular_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, hessenberg};
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
