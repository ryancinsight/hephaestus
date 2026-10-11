#[cfg(feature = "decomposition")]
#[test]
fn qr_rejects_underdetermined_rank_deficient_and_nonfinite_inputs() {
    let Some(device) = device("qr_rejects_underdetermined_rank_deficient_and_nonfinite_inputs")
    else {
        return;
    };

    let underdetermined_buffer = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("HIP underdetermined QR upload");
    let underdetermined_layout = Layout::c_contiguous([2, 2]).expect("square QR layout");
    let rank_deficient_buffer = device
        .upload(&[1.0_f32, 2.0, 2.0, 4.0, 3.0, 6.0])
        .expect("HIP rank-deficient QR upload");
    let rank_deficient_layout = Layout::c_contiguous([3, 2]).expect("rank-deficient QR layout");
    assert!(matches!(
        qr_decompose(
            &device,
            StridedOperand {
                buffer: &underdetermined_buffer,
                layout: &underdetermined_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("m ≥ n")
    ));
    assert!(matches!(
        qr_decompose(
            &device,
            StridedOperand {
                buffer: &rank_deficient_buffer,
                layout: &rank_deficient_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("rank-deficient")
    ));

    let nonfinite_buffer = device
        .upload(&[1.0_f32, f32::NAN, 3.0, 4.0, 5.0, 6.0])
        .expect("HIP nonfinite QR upload");
    assert!(matches!(
        qr_decompose(
            &device,
            StridedOperand {
                buffer: &nonfinite_buffer,
                layout: &rank_deficient_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("non-finite")
    ));
}

#[cfg(feature = "decomposition")]
#[test]
fn svd_surfaces_match_leto_values_and_rank_contracts() {
    let Some(device) = device("svd_surfaces_match_leto_values_and_rank_contracts") else {
        return;
    };

    let matrix_values = [3.0_f32, 0.0, 0.0, 4.0, 0.0, 0.0];
    let matrix = device.upload(&matrix_values).expect("HIP SVD upload");
    let layout = Layout::c_contiguous([3, 2]).expect("SVD layout");
    let expected = leto_ops::svd_decompose(&leto::ArrayView::<f32, 2>::new(layout, &matrix_values))
        .expect("CPU SVD reference");
    let factor = svd_decompose(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm thin SVD");
    assert_eq!(factor.shape(), (3, 2));
    let mut actual_values = vec![0.0_f32; expected.singular_values.len()];
    device
        .download(factor.singular_values(), &mut actual_values)
        .expect("HIP thin SVD singular values download");
    for (actual, expected) in actual_values.iter().zip(&expected.singular_values) {
        assert_near(*actual, *expected, 2048.0);
    }
    let mut actual_u =
        vec![0.0_f32; leto::Storage::as_slice(expected.left_singular_vectors.storage()).len()];
    let mut actual_v =
        vec![0.0_f32; leto::Storage::as_slice(expected.right_singular_vectors.storage()).len()];
    device
        .download(factor.u(), &mut actual_u)
        .expect("HIP thin SVD U download");
    device
        .download(factor.v(), &mut actual_v)
        .expect("HIP thin SVD V download");
    let reconstructed = reconstruct_svd(&actual_u, &actual_v, &actual_values, 3, 2);
    for (actual, expected) in reconstructed.iter().zip(matrix_values) {
        assert_near(*actual, expected, 4096.0);
    }

    let only_values = singular_values(
        &device,
        StridedOperand {
            buffer: &matrix,
            layout: &layout,
        },
    )
    .expect("ROCm singular-values-only SVD");
    let mut only_values_host = vec![0.0_f32; expected.singular_values.len()];
    device
        .download(&only_values, &mut only_values_host)
        .expect("HIP singular-values-only download");
    for (actual, expected) in only_values_host.iter().zip(&expected.singular_values) {
        assert_near(*actual, *expected, 2048.0);
    }

    let rank_deficient_values = [1.0_f32, 2.0, 2.0, 4.0, 3.0, 6.0];
    let rank_deficient = device
        .upload(&rank_deficient_values)
        .expect("HIP rank-deficient SVD upload");
    let rank_deficient_factor = svd_rank_revealing(
        &device,
        StridedOperand {
            buffer: &rank_deficient,
            layout: &layout,
        },
    )
    .expect("ROCm rank-revealing SVD");
    let rank_expected = leto_ops::svd_decompose(&leto::ArrayView::<f32, 2>::new(
        layout,
        &rank_deficient_values,
    ))
    .expect("CPU rank-revealing SVD reference");
    let mut rank_values = vec![0.0_f32; rank_expected.singular_values.len()];
    device
        .download(rank_deficient_factor.singular_values(), &mut rank_values)
        .expect("HIP rank-revealing values download");
    for (actual, expected) in rank_values.iter().zip(&rank_expected.singular_values) {
        assert_near(*actual, *expected, 4096.0);
    }
    let mut rank_u =
        vec![0.0_f32; leto::Storage::as_slice(rank_expected.left_singular_vectors.storage()).len()];
    let mut rank_v = vec![
        0.0_f32;
        leto::Storage::as_slice(rank_expected.right_singular_vectors.storage())
            .len()
    ];
    device
        .download(rank_deficient_factor.u(), &mut rank_u)
        .expect("HIP rank-revealing U download");
    device
        .download(rank_deficient_factor.v(), &mut rank_v)
        .expect("HIP rank-revealing V download");
    let reconstructed = reconstruct_svd(&rank_u, &rank_v, &rank_values, 3, 2);
    for (actual, expected) in reconstructed.iter().zip(rank_deficient_values) {
        assert_near(*actual, expected, 4096.0);
    }
    assert!(matches!(
        svd_decompose(
            &device,
            StridedOperand {
                buffer: &rank_deficient,
                layout: &layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message }) if message.contains("thin SVD")
    ));

    let empty = device.upload(&[] as &[f32]).expect("HIP empty SVD upload");
    let empty_layout = Layout::c_contiguous([0, 0]).expect("empty SVD layout");
    let empty_factor = svd_decompose(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("ROCm empty SVD");
    assert_eq!(empty_factor.shape(), (0, 0));
}
