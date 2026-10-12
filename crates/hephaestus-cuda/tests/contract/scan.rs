use super::*;

#[test]
fn scan_scan_axis_matches_cpu() {
    let Some(dev) = device("scan_scan_axis_matches_cpu") else {
        return;
    };

    let host_in = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = dev.upload(&host_in).unwrap();
    let a_layout = Layout::c_contiguous([2, 2]).unwrap();

    let out = scan_axis::<_, CumSumOp, f32>(
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
fn scan_suffix_sum_matches_leto_for_allocated_and_caller_owned_outputs() {
    let Some(dev) = device("scan_suffix_sum_matches_leto_for_allocated_and_caller_owned_outputs")
    else {
        return;
    };

    let host = vec![1i32, 2, 3, 4, 5, 6];
    let input = dev.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let expected = leto_ops::scan_axis::<leto_ops::CumSumOp, _, 2>(
        &leto::Array::from_shape_vec([2, 3], host).unwrap().view(),
        1,
        leto_ops::ScanDirection::Reverse,
    )
    .unwrap()
    .into_vec();

    let allocated = suffix_sum(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_allocated = vec![0i32; expected.len()];
    dev.download(&allocated, &mut got_allocated).unwrap();
    assert_eq!(got_allocated, expected);

    let output = dev.alloc_zeroed::<i32>(expected.len()).unwrap();
    suffix_sum_into(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
        1,
        StridedOperand {
            buffer: &output,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_into = vec![0i32; expected.len()];
    dev.download(&output, &mut got_into).unwrap();
    assert_eq!(got_into, expected);
}

#[test]
fn scan_cumprod_convenience_preserves_strided_and_empty_contract() {
    let Some(dev) = device("scan_cumprod_convenience_preserves_strided_and_empty_contract") else {
        return;
    };

    let physical = vec![1_i32, 2, 3, 4, 5, 6];
    let input = dev.upload(&physical).unwrap();
    let transposed_layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let output_layout = Layout::c_contiguous([2, 3]).unwrap();
    let output = dev.alloc_zeroed::<i32>(6).unwrap();
    cumprod_into(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = [0_i32; 6];
    dev.download(&output, &mut got).unwrap();
    assert_eq!(got, [1, 3, 15, 2, 8, 48]);

    let allocated = cumprod(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_allocated = [0_i32; 6];
    dev.download(&allocated, &mut got_allocated).unwrap();
    assert_eq!(got_allocated, [1, 3, 5, 2, 12, 30]);

    let suffix = suffix_prod(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix = [0_i32; 6];
    dev.download(&suffix, &mut got_suffix).unwrap();
    assert_eq!(got_suffix, [15, 15, 5, 48, 24, 6]);

    let suffix_into = dev.alloc_zeroed::<i32>(6).unwrap();
    suffix_prod_into(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        StridedOperand {
            buffer: &suffix_into,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_into = [0_i32; 6];
    dev.download(&suffix_into, &mut got_suffix_into).unwrap();
    assert_eq!(got_suffix_into, [2, 12, 30, 2, 4, 6]);

    let empty = dev.alloc_zeroed::<i32>(0).unwrap();
    let empty_layout = Layout::c_contiguous([2, 0]).unwrap();
    let empty_output = cumprod(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    assert_eq!(empty_output.len(), 0);

    let invalid_layout = Layout::try_new([2, 3], [1, 2], 1).expect("valid test layout");
    assert!(matches!(
        cumprod(
            &dev,
            StridedOperand {
                buffer: &input,
                layout: &invalid_layout,
            },
            1,
            BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}

#[test]
fn scan_long_line_matches_integer_reference() {
    let Some(dev) = device("scan_long_line_matches_integer_reference") else {
        return;
    };

    let cols = 513usize;
    let host: Vec<i32> = (0..2 * cols)
        .map(|index| i32::try_from(index).expect("test index fits i32") - 300)
        .collect();
    let input = dev.upload(&host).unwrap();
    let layout = Layout::c_contiguous([2, cols]).unwrap();
    let output = scan_axis::<_, CumSumOp, i32>(
        &dev,
        StridedOperand {
            buffer: &input,
            layout: &layout,
        },
        1,
        hephaestus_cuda::ScanDirection::Forward,
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0i32; host.len()];
    dev.download(&output, &mut got).unwrap();
    let expected: Vec<i32> = host
        .chunks_exact(cols)
        .flat_map(|row| {
            let mut acc = 0i32;
            row.iter().map(move |value| {
                acc += *value;
                acc
            })
        })
        .collect();
    assert_eq!(got, expected);
}
