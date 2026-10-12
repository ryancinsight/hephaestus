use super::*;

#[test]
fn scan_kernels_match_cpu_values_across_axes_directions_and_chunk_boundaries() {
    let Some(device) =
        device("scan_kernels_match_cpu_values_across_axes_directions_and_chunk_boundaries")
    else {
        return;
    };

    let width = BlockWidth::new(2).expect("test scan width is non-zero");
    let input: Vec<i32> = (1..=8).collect();
    let input_buffer = device.upload(&input).expect("HIP scan input upload");
    let input_layout = Layout::c_contiguous([2, 4]).expect("scan input layout");
    let input_operand = StridedOperand {
        buffer: &input_buffer,
        layout: &input_layout,
    };

    let forward = cumsum(&device, input_operand, 1, width).expect("HIP forward cumulative sum");
    let mut forward_output = [0_i32; 8];
    device
        .download(&forward, &mut forward_output)
        .expect("HIP forward scan download");
    assert_eq!(forward_output, [1, 3, 6, 10, 5, 11, 18, 26]);

    let reverse =
        scan_axis::<_, CumSumOp, _>(&device, input_operand, 1, ScanDirection::Reverse, width)
            .expect("HIP reverse cumulative sum");
    let mut reverse_output = [0_i32; 8];
    device
        .download(&reverse, &mut reverse_output)
        .expect("HIP reverse scan download");
    assert_eq!(reverse_output, [10, 9, 7, 4, 26, 21, 15, 8]);

    let suffix = suffix_sum(&device, input_operand, 1, width).expect("HIP suffix sum");
    let mut suffix_output = [0_i32; 8];
    device
        .download(&suffix, &mut suffix_output)
        .expect("HIP suffix sum download");
    assert_eq!(suffix_output, reverse_output);

    let suffix_into = device
        .alloc_zeroed::<i32>(8)
        .expect("HIP caller-owned suffix output");
    suffix_sum_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &suffix_into,
            layout: &input_layout,
        },
        width,
    )
    .expect("HIP caller-owned suffix sum");
    let mut suffix_into_output = [0_i32; 8];
    device
        .download(&suffix_into, &mut suffix_into_output)
        .expect("HIP caller-owned suffix download");
    assert_eq!(suffix_into_output, [6, 8, 10, 12, 5, 6, 7, 8]);

    let column_scan = cumsum(&device, input_operand, 0, width).expect("HIP column scan");
    let mut column_output = [0_i32; 8];
    device
        .download(&column_scan, &mut column_output)
        .expect("HIP column scan download");
    assert_eq!(column_output, [1, 2, 3, 4, 6, 8, 10, 12]);

    let forward_product =
        cumprod(&device, input_operand, 1, width).expect("HIP forward cumulative product");
    let mut product_output = [0_i32; 8];
    device
        .download(&forward_product, &mut product_output)
        .expect("HIP product scan download");
    assert_eq!(product_output, [1, 2, 6, 24, 5, 30, 210, 1680]);

    let product_into = device
        .alloc_zeroed::<i32>(8)
        .expect("HIP caller-owned product output");
    cumprod_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &product_into,
            layout: &input_layout,
        },
        width,
    )
    .expect("HIP caller-owned forward product scan");
    let mut product_into_output = [0_i32; 8];
    device
        .download(&product_into, &mut product_into_output)
        .expect("HIP caller-owned product download");
    assert_eq!(product_into_output, [1, 2, 3, 4, 5, 12, 21, 32]);

    let reverse_product =
        suffix_prod(&device, input_operand, 1, width).expect("HIP reverse cumulative product");
    let mut reverse_product_output = [0_i32; 8];
    device
        .download(&reverse_product, &mut reverse_product_output)
        .expect("HIP reverse product download");
    assert_eq!(reverse_product_output, [24, 24, 12, 4, 1680, 336, 56, 8]);

    let reverse_product_into = device
        .alloc_zeroed::<i32>(8)
        .expect("HIP caller-owned reverse product output");
    suffix_prod_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &reverse_product_into,
            layout: &input_layout,
        },
        width,
    )
    .expect("HIP caller-owned reverse product scan");
    let mut reverse_product_into_output = [0_i32; 8];
    device
        .download(&reverse_product_into, &mut reverse_product_into_output)
        .expect("HIP caller-owned reverse product download");
    assert_eq!(reverse_product_into_output, [5, 12, 21, 32, 5, 6, 7, 8]);

    let long_input: Vec<i32> = (0..1_025).map(|index| index % 7 - 3).collect();
    let long_buffer = device.upload(&long_input).expect("HIP long scan upload");
    let long_layout = Layout::c_contiguous([1, 1_025]).expect("long scan layout");
    let long_output = scan_axis::<_, CumSumOp, _>(
        &device,
        StridedOperand {
            buffer: &long_buffer,
            layout: &long_layout,
        },
        1,
        ScanDirection::Forward,
        width,
    )
    .expect("HIP long scan");
    let mut got_long = vec![0_i32; long_input.len()];
    device
        .download(&long_output, &mut got_long)
        .expect("HIP long scan download");
    let expected_long: Vec<i32> = long_input
        .iter()
        .scan(0_i32, |acc, value| {
            *acc += *value;
            Some(*acc)
        })
        .collect();
    assert_eq!(got_long, expected_long);

    let wrong_layout = Layout::c_contiguous([4, 2]).expect("wrong scan output layout");
    let wrong_buffer = device
        .alloc_zeroed::<i32>(8)
        .expect("wrong scan output buffer");
    assert!(matches!(
        scan_axis_into::<_, CumSumOp, _>(
            &device,
            input_operand,
            1,
            ScanDirection::Forward,
            StridedOperand {
                buffer: &wrong_buffer,
                layout: &wrong_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("scan output shape mismatch")
    ));

    assert!(matches!(
        scan_axis_into::<_, CumSumOp, _>(
            &device,
            input_operand,
            1,
            ScanDirection::Forward,
            StridedOperand {
                buffer: &input_buffer,
                layout: &input_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "scan output buffer must not alias input buffer"
    ));
}
