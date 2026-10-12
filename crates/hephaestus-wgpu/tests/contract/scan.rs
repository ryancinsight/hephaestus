use super::*;

pub(crate) fn axis_scans_match_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{
        StridedOperand, cumsum, suffix_prod, suffix_prod_into, suffix_sum, suffix_sum_into,
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

    let suffix_sum_axis1 = device.alloc_zeroed::<i32>(6).unwrap();
    suffix_sum_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &suffix_sum_axis1,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_suffix_sum_axis1 = leto_ops::scan_axis::<leto_ops::CumSumOp, _, 2>(
        &leto_input.view(),
        1,
        leto_ops::ScanDirection::Reverse,
    )
    .unwrap()
    .into_vec();
    let mut got_suffix_sum_axis1 = vec![0i32; 6];
    device
        .download(&suffix_sum_axis1, &mut got_suffix_sum_axis1)
        .unwrap();
    assert_eq!(got_suffix_sum_axis1, expected_suffix_sum_axis1);

    let suffix_sum_allocated = suffix_sum(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_suffix_sum_allocated = vec![0i32; 6];
    device
        .download(&suffix_sum_allocated, &mut got_suffix_sum_allocated)
        .unwrap();
    let expected_suffix_sum_axis0 = leto_ops::scan_axis::<leto_ops::CumSumOp, _, 2>(
        &leto_input.view(),
        0,
        leto_ops::ScanDirection::Reverse,
    )
    .unwrap()
    .into_vec();
    assert_eq!(got_suffix_sum_allocated, expected_suffix_sum_axis0);

    let suffix_prod_reverse = device.alloc_zeroed::<i32>(6).unwrap();
    suffix_prod_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &suffix_prod_reverse,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_suffix_prod_reverse = leto_ops::scan_axis::<leto_ops::CumProdOp, _, 2>(
        &leto_input.view(),
        0,
        leto_ops::ScanDirection::Reverse,
    )
    .unwrap()
    .into_vec();
    let mut got_suffix_prod_reverse = vec![0i32; 6];
    device
        .download(&suffix_prod_reverse, &mut got_suffix_prod_reverse)
        .unwrap();
    assert_eq!(got_suffix_prod_reverse, expected_suffix_prod_reverse);

    let suffix_prod_reverse_allocated =
        suffix_prod(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_suffix_prod_reverse_allocated = vec![0i32; 6];
    device
        .download(
            &suffix_prod_reverse_allocated,
            &mut got_suffix_prod_reverse_allocated,
        )
        .unwrap();
    assert_eq!(
        got_suffix_prod_reverse_allocated,
        expected_suffix_prod_reverse
    );
}

pub(crate) fn cumprod_convenience_preserves_strided_and_empty_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };

    use hephaestus_wgpu::{StridedOperand, cumprod, cumprod_into};
    use leto::Layout;

    let physical = vec![1_i32, 2, 3, 4, 5, 6];
    let input = device.upload(&physical).unwrap();
    let transposed_layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let output_layout = Layout::c_contiguous([2, 3]).unwrap();
    let output = device.alloc_zeroed::<i32>(6).unwrap();
    cumprod_into(
        &device,
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
    device.download(&output, &mut got).unwrap();
    assert_eq!(got, [1, 3, 15, 2, 8, 48]);

    let allocated = cumprod(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_allocated = [0_i32; 6];
    device.download(&allocated, &mut got_allocated).unwrap();
    assert_eq!(got_allocated, [1, 3, 5, 2, 12, 30]);

    let empty = device.alloc_zeroed::<i32>(0).unwrap();
    let empty_layout = Layout::c_contiguous([2, 0]).unwrap();
    let empty_output = cumprod(
        &device,
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
            &device,
            StridedOperand {
                buffer: &input,
                layout: &invalid_layout,
            },
            1,
            BlockWidth::DEFAULT,
        ),
        Err(hephaestus_core::HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}
