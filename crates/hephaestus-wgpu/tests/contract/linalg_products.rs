use super::*;

pub(crate) fn linalg_matmul_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, matmul, matmul_into};
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

pub(crate) fn linalg_batched_matmul_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, batched_matmul, batched_matmul_into};
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

pub(crate) fn linalg_allocating_kron_matches_the_into_path() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, kron, kron_into};
    use leto::Layout;

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_host = vec![7.0f32, 8.0, 9.0, 10.0];

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let a_layout = Layout::c_contiguous([2, 3]).unwrap();
    let b_layout = Layout::c_contiguous([2, 2]).unwrap();
    let out_layout = Layout::c_contiguous([4, 6]).unwrap();

    // The seam (`kron_into`) owns the correctness contract — the shared
    // conformance clause pins it exactly, on every backend. What is
    // backend-specific here is the allocating wrapper: it must agree with
    // the seam element-for-element rather than re-deriving anything.
    let out = device.alloc_zeroed::<f32>(24).unwrap();
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
    let mut into_got = vec![0.0f32; 24];
    device.download(&out, &mut into_got).unwrap();

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
    let mut alloc_got = vec![0.0f32; 24];
    device.download(&allocated, &mut alloc_got).unwrap();
    assert_eq!(
        alloc_got, into_got,
        "the allocating kron wrapper must match kron_into element-for-element"
    );
}

pub(crate) fn linalg_matpow_zero_exponent_overwrites_alternate_identity_storage() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, matpow};
    use leto::Layout;

    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let alternate = device.upload(&[AlternateIdentity(4); 4]).unwrap();
    let identity = matpow(
        &device,
        StridedOperand {
            buffer: &alternate,
            layout: &layout,
        },
        0,
    )
    .unwrap();
    let mut actual = vec![AlternateIdentity(0); 4];
    device.download(&identity, &mut actual).unwrap();
    assert_eq!(
        actual,
        vec![
            AlternateIdentity(9),
            AlternateIdentity(-7),
            AlternateIdentity(-7),
            AlternateIdentity(9),
        ]
    );
}

pub(crate) fn linalg_matpow_rejects_non_square() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, matpow};
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

pub(crate) fn linalg_dot_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, dot};
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

pub(crate) fn linalg_trace_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, trace};
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

pub(crate) fn linalg_det_rejects_rectangular_operands() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, det};
    use leto::Layout;

    let rectangular = device.alloc_zeroed::<f32>(6).unwrap();
    let rectangular_layout = Layout::c_contiguous([2, 3]).unwrap();
    let result = det(
        &device,
        StridedOperand {
            buffer: &rectangular,
            layout: &rectangular_layout,
        },
    );
    assert!(matches!(
        result,
        Err(HephaestusError::DispatchFailed { message }) if message.contains("square matrix")
    ));
}

#[cfg(feature = "decomposition")]
pub(crate) fn bidiagonalize_rejects_wide_matrix() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, bidiagonalize};
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
