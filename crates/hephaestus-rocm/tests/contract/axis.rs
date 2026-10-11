use super::*;

#[test]
fn prepared_axis_reductions_reuse_plans_and_validate_contracts() {
    let Some(device) = device("prepared_axis_reductions_reuse_plans_and_validate_contracts") else {
        return;
    };

    let host: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let input = device
        .upload(&host)
        .expect("HIP prepared axis input upload");
    let input_layout = Layout::c_contiguous([3, 4]).expect("prepared axis input layout");
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let width = BlockWidth::new(2).expect("prepared axis width");

    let axis0_out = device
        .alloc_zeroed::<f32>(4)
        .expect("HIP prepared axis output");
    let axis0_layout = Layout::c_contiguous([1, 4]).expect("prepared axis output layout");
    let prepared_sum_axis0 = prepare_sum_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        width,
    )
    .expect("HIP prepared column sum");
    prepared_sum_axis0
        .dispatch(&device)
        .expect("HIP prepared column sum dispatch");
    let mut got_axis0 = [0.0f32; 4];
    device
        .download(&axis0_out, &mut got_axis0)
        .expect("HIP prepared column sum download");
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);
    prepared_sum_axis0
        .dispatch(&device)
        .expect("HIP repeated column sum dispatch");
    device
        .download(&axis0_out, &mut got_axis0)
        .expect("HIP repeated column sum download");
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);

    let transposed_layout = Layout::try_new([4, 3], [1, 4], 0).expect("valid test layout");
    let transposed_input = StridedOperand {
        buffer: &input,
        layout: &transposed_layout,
    };
    let axis1_out = device
        .alloc_zeroed::<f32>(4)
        .expect("HIP prepared transposed output");
    let axis1_layout = Layout::c_contiguous([4, 1]).expect("prepared transposed output layout");
    let prepared_sum_axis1 = prepare_sum_axis_into(
        &device,
        transposed_input,
        1,
        StridedOperand {
            buffer: &axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .expect("HIP prepared transposed sum");
    let max_axis0_out = device
        .alloc_zeroed::<f32>(3)
        .expect("HIP prepared transposed max output");
    let max_axis0_layout = Layout::c_contiguous([1, 3]).expect("prepared transposed max layout");
    let prepared_max_axis0 = prepare_max_axis_into(
        &device,
        transposed_input,
        0,
        StridedOperand {
            buffer: &max_axis0_out,
            layout: &max_axis0_layout,
        },
        width,
    )
    .expect("HIP prepared transposed max");
    submit_prepared_axis_reduction_batch(&device, &[&prepared_sum_axis1, &prepared_max_axis0])
        .expect("HIP prepared axis batch");
    let mut got_axis1 = [0.0f32; 4];
    let mut got_max_axis0 = [0.0f32; 3];
    device
        .download(&axis1_out, &mut got_axis1)
        .expect("HIP prepared transposed sum download");
    device
        .download(&max_axis0_out, &mut got_max_axis0)
        .expect("HIP prepared transposed max download");
    assert_eq!(got_axis1, [15.0, 18.0, 21.0, 24.0]);
    assert_eq!(got_max_axis0, [4.0, 8.0, 12.0]);

    let mean_axis1_out = device
        .alloc_zeroed::<f32>(4)
        .expect("HIP prepared mean output");
    let prepared_mean_axis1 = prepare_mean_axis_into(
        &device,
        transposed_input,
        1,
        StridedOperand {
            buffer: &mean_axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .expect("HIP prepared mean");
    prepared_mean_axis1
        .dispatch(&device)
        .expect("HIP prepared mean dispatch");
    let mut got_mean_axis1 = [0.0f32; 4];
    device
        .download(&mean_axis1_out, &mut got_mean_axis1)
        .expect("HIP prepared mean download");
    assert_eq!(got_mean_axis1, [5.0, 6.0, 7.0, 8.0]);

    let empty_input = device.upload::<f32>(&[]).expect("HIP prepared empty input");
    let empty_input_layout = Layout::c_contiguous([3, 0]).expect("prepared empty input layout");
    let empty_output = device
        .upload(&[7.0f32; 3])
        .expect("HIP prepared empty output");
    let empty_output_layout = Layout::c_contiguous([3, 1]).expect("prepared empty output layout");
    let prepared_empty_sum = prepare_sum_axis_into(
        &device,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    )
    .expect("HIP prepared empty sum");
    prepared_empty_sum
        .dispatch(&device)
        .expect("HIP prepared empty sum dispatch");
    let mut got_empty = [7.0f32; 3];
    device
        .download(&empty_output, &mut got_empty)
        .expect("HIP prepared empty sum download");
    assert_eq!(got_empty, [0.0, 0.0, 0.0]);

    let empty_product = prod_axis(
        &device,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        width,
    )
    .expect("HIP empty product");
    let mut got_empty_product = [0.0f32; 3];
    device
        .download(&empty_product, &mut got_empty_product)
        .expect("HIP empty product download");
    assert_eq!(got_empty_product, [1.0, 1.0, 1.0]);

    let empty_mean = prepare_mean_axis_into(
        &device,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    );
    assert!(matches!(
        empty_mean,
        Err(HephaestusError::DispatchFailed { message })
            if message == "mean_axis is undefined for empty axis 1"
    ));

    let alias_layout = Layout::c_contiguous([3, 1]).expect("prepared alias layout");
    let alias = prepare_sum_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &input,
            layout: &alias_layout,
        },
        width,
    );
    assert!(matches!(
        alias,
        Err(HephaestusError::DispatchFailed { message })
            if message == "axis reduction output buffer must not alias input buffer"
    ));

    let invalid_width = BlockWidth::new(3).expect("prepared invalid width");
    let invalid = prepare_sum_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        invalid_width,
    );
    assert!(matches!(
        invalid,
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 3 must be a power of two"
    ));
}
