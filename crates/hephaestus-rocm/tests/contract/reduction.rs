use super::*;

#[test]
fn reduction_kernels_match_cpu_values_across_tree_passes_and_boundaries() {
    let Some(device) =
        device("reduction_kernels_match_cpu_values_across_tree_passes_and_boundaries")
    else {
        return;
    };

    let input: Vec<u32> = (0..513).map(|index| (index % 17) as u32).collect();
    let input_buffer = device.upload(&input).expect("HIP reduction input upload");
    let width = BlockWidth::new(128).expect("test reduction width is non-zero");

    let sum =
        reduction_with_width::<SumOp, _>(&device, &input_buffer, width).expect("HIP sum reduction");
    let min =
        reduction_with_width::<MinOp, _>(&device, &input_buffer, width).expect("HIP min reduction");
    let max =
        reduction_with_width::<MaxOp, _>(&device, &input_buffer, width).expect("HIP max reduction");

    let expected_sum: u32 = input.iter().copied().sum();
    let expected_min = input.iter().copied().min().expect("non-empty input");
    let expected_max = input.iter().copied().max().expect("non-empty input");
    let mut sum_output = [0_u32];
    let mut min_output = [0_u32];
    let mut max_output = [0_u32];
    device
        .download(&sum, &mut sum_output)
        .expect("HIP sum download");
    device
        .download(&min, &mut min_output)
        .expect("HIP min download");
    device
        .download(&max, &mut max_output)
        .expect("HIP max download");
    assert_eq!(sum_output, [expected_sum]);
    assert_eq!(min_output, [expected_min]);
    assert_eq!(max_output, [expected_max]);

    let empty = device
        .upload::<u32>(&[])
        .expect("HIP empty reduction upload");
    let empty_sum =
        reduction_with_width::<SumOp, _>(&device, &empty, width).expect("HIP empty sum reduction");
    let mut empty_output = [u32::MAX];
    device
        .download(&empty_sum, &mut empty_output)
        .expect("HIP empty sum download");
    assert_eq!(empty_output, [0]);

    let invalid_width = BlockWidth::new(192).expect("test invalid width is non-zero");
    assert!(matches!(
        reduction_with_width::<SumOp, _>(&device, &input_buffer, invalid_width),
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 192 must be a power of two"
    ));
}

#[test]
fn axis_reduction_kernels_match_cpu_values_and_reject_invalid_layouts() {
    let Some(device) = device("axis_reduction_kernels_match_cpu_values_and_reject_invalid_layouts")
    else {
        return;
    };

    let width = BlockWidth::new(2).expect("test reduction width is non-zero");
    let input: Vec<u32> = vec![1, 2, 3, 4, 10, 20, 30, 40, 5, 7, 9, 11];
    let input_buffer = device.upload(&input).expect("HIP axis input upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("axis input layout");
    let input_operand = StridedOperand {
        buffer: &input_buffer,
        layout: &input_layout,
    };

    let sum_rows = sum_axis(&device, input_operand, 1, width).expect("HIP axis row sum");
    let sum_columns = sum_axis(&device, input_operand, 0, width).expect("HIP axis column sum");
    let min_rows = min_axis(&device, input_operand, 1, width).expect("HIP axis row min");
    let max_rows = max_axis(&device, input_operand, 1, width).expect("HIP axis row max");
    let mut sum_rows_output = [0_u32; 3];
    let mut sum_columns_output = [0_u32; 4];
    let mut min_rows_output = [0_u32; 3];
    let mut max_rows_output = [0_u32; 3];
    device
        .download(&sum_rows, &mut sum_rows_output)
        .expect("HIP row sum download");
    device
        .download(&sum_columns, &mut sum_columns_output)
        .expect("HIP column sum download");
    device
        .download(&min_rows, &mut min_rows_output)
        .expect("HIP row min download");
    device
        .download(&max_rows, &mut max_rows_output)
        .expect("HIP row max download");
    assert_eq!(sum_rows_output, [10, 100, 32]);
    assert_eq!(sum_columns_output, [16, 29, 42, 55]);
    assert_eq!(min_rows_output, [1, 10, 5]);
    assert_eq!(max_rows_output, [4, 40, 11]);

    let product_rows = prod_axis(&device, input_operand, 1, width).expect("HIP axis row product");
    let mut product_rows_output = [0_u32; 3];
    device
        .download(&product_rows, &mut product_rows_output)
        .expect("HIP row product download");
    assert_eq!(product_rows_output, [24, 240_000, 3465]);

    let product_columns = reduce_axis::<ProdOp, u32>(&device, input_operand, 0, width)
        .expect("HIP axis column product");
    let mut product_columns_output = [0_u32; 4];
    device
        .download(&product_columns, &mut product_columns_output)
        .expect("HIP column product download");
    assert_eq!(product_columns_output, [50, 280, 810, 17_600]);

    let product_rows_into = device
        .alloc_zeroed::<u32>(3)
        .expect("HIP caller-owned product output");
    let product_rows_into_layout =
        Layout::c_contiguous([3, 1]).expect("HIP caller-owned product layout");
    prod_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &product_rows_into,
            layout: &product_rows_into_layout,
        },
        width,
    )
    .expect("HIP caller-owned row product");
    let mut product_rows_into_output = [0_u32; 3];
    device
        .download(&product_rows_into, &mut product_rows_into_output)
        .expect("HIP caller-owned product download");
    assert_eq!(product_rows_into_output, [24, 240_000, 3465]);

    let transposed_layout = Layout::try_new([4, 3], [1, 4], 0).expect("valid test layout");
    let transposed_product = prod_axis(
        &device,
        StridedOperand {
            buffer: &input_buffer,
            layout: &transposed_layout,
        },
        1,
        width,
    )
    .expect("HIP strided product");
    let mut transposed_product_output = [0_u32; 4];
    device
        .download(&transposed_product, &mut transposed_product_output)
        .expect("HIP strided product download");
    assert_eq!(transposed_product_output, [50, 280, 810, 17_600]);

    let mean_input: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let mean_buffer = device.upload(&mean_input).expect("HIP mean input upload");
    let mean_layout = Layout::c_contiguous([3, 4]).expect("mean input layout");
    let mean = mean_axis(
        &device,
        StridedOperand {
            buffer: &mean_buffer,
            layout: &mean_layout,
        },
        1,
        width,
    )
    .expect("HIP row mean");
    let mut mean_output = [0.0_f32; 3];
    device
        .download(&mean, &mut mean_output)
        .expect("HIP mean download");
    assert_eq!(mean_output, [2.5, 6.5, 10.5]);

    let wrong_layout = Layout::c_contiguous([3, 2]).expect("wrong output layout");
    let wrong_buffer = device.alloc_zeroed::<u32>(6).expect("wrong output buffer");
    assert!(matches!(
        hephaestus_rocm::sum_axis_into(
            &device,
            input_operand,
            1,
            StridedOperand {
                buffer: &wrong_buffer,
                layout: &wrong_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("axis reduction output shape mismatch")
    ));

    assert!(matches!(
        prod_axis_into(
            &device,
            input_operand,
            1,
            StridedOperand {
                buffer: &wrong_buffer,
                layout: &wrong_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("axis reduction output shape mismatch")
    ));

    let output_layout = Layout::c_contiguous([3, 1]).expect("alias output layout");
    assert!(matches!(
        hephaestus_rocm::sum_axis_into(
            &device,
            input_operand,
            1,
            StridedOperand {
                buffer: &input_buffer,
                layout: &output_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "axis reduction output buffer must not alias input buffer"
    ));

    let empty_buffer = device.upload::<u32>(&[]).expect("empty axis input upload");
    let empty_input_layout = Layout::c_contiguous([3, 0]).expect("empty input layout");
    let empty_output_layout = Layout::c_contiguous([3, 1]).expect("empty output layout");
    let empty_output = device.alloc_zeroed::<u32>(3).expect("empty output buffer");
    assert!(matches!(
        mean_axis_into(
            &device,
            StridedOperand {
                buffer: &empty_buffer,
                layout: &empty_input_layout,
            },
            1,
            StridedOperand {
                buffer: &empty_output,
                layout: &empty_output_layout,
            },
            width,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message == "mean_axis is undefined for empty axis 1"
    ));

    let empty_axis0_input_layout = Layout::c_contiguous([0, 3]).expect("empty axis-0 layout");
    let empty_axis0_product = prod_axis(
        &device,
        StridedOperand {
            buffer: &empty_buffer,
            layout: &empty_axis0_input_layout,
        },
        0,
        width,
    )
    .expect("HIP empty axis-0 product");
    let mut got_empty_axis0_product = [0_u32; 3];
    device
        .download(&empty_axis0_product, &mut got_empty_axis0_product)
        .expect("HIP empty axis-0 product download");
    assert_eq!(got_empty_axis0_product, [1, 1, 1]);
}
