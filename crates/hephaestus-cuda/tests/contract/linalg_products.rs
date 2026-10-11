use super::*;

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
