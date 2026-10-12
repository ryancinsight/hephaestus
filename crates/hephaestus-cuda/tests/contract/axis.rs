use super::*;

#[test]
fn prepared_axis_reductions_reuse_plans_and_validate_contracts() {
    let Some(dev) = device("prepared_axis_reductions_reuse_plans_and_validate_contracts") else {
        return;
    };

    let host: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let input = dev.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([3, 4]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let width = BlockWidth::new(2).unwrap();

    let axis0_out = dev.alloc_zeroed::<f32>(4).unwrap();
    let axis0_layout = Layout::c_contiguous([1, 4]).unwrap();
    let prepared_sum_axis0 = prepare_sum_axis_into(
        &dev,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        width,
    )
    .unwrap();
    prepared_sum_axis0.dispatch(&dev).unwrap();
    let mut got_axis0 = [0.0f32; 4];
    dev.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);
    prepared_sum_axis0.dispatch(&dev).unwrap();
    dev.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);

    let transposed_layout = Layout::try_new([4, 3], [1, 4], 0).expect("valid test layout");
    let transposed_input = StridedOperand {
        buffer: &input,
        layout: &transposed_layout,
    };
    let axis1_out = dev.alloc_zeroed::<f32>(4).unwrap();
    let axis1_layout = Layout::c_contiguous([4, 1]).unwrap();
    let prepared_sum_axis1 = prepare_sum_axis_into(
        &dev,
        transposed_input,
        1,
        StridedOperand {
            buffer: &axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    let max_axis0_out = dev.alloc_zeroed::<f32>(3).unwrap();
    let max_axis0_layout = Layout::c_contiguous([1, 3]).unwrap();
    let prepared_max_axis0 = prepare_max_axis_into(
        &dev,
        transposed_input,
        0,
        StridedOperand {
            buffer: &max_axis0_out,
            layout: &max_axis0_layout,
        },
        width,
    )
    .unwrap();
    submit_prepared_axis_reduction_batch(&dev, &[&prepared_sum_axis1, &prepared_max_axis0])
        .unwrap();
    let mut got_axis1 = [0.0f32; 4];
    let mut got_max_axis0 = [0.0f32; 3];
    dev.download(&axis1_out, &mut got_axis1).unwrap();
    dev.download(&max_axis0_out, &mut got_max_axis0).unwrap();
    assert_eq!(got_axis1, [15.0, 18.0, 21.0, 24.0]);
    assert_eq!(got_max_axis0, [4.0, 8.0, 12.0]);

    let mean_axis1_out = dev.alloc_zeroed::<f32>(4).unwrap();
    let prepared_mean_axis1 = prepare_mean_axis_into(
        &dev,
        transposed_input,
        1,
        StridedOperand {
            buffer: &mean_axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    prepared_mean_axis1.dispatch(&dev).unwrap();
    let mut got_mean_axis1 = [0.0f32; 4];
    dev.download(&mean_axis1_out, &mut got_mean_axis1).unwrap();
    assert_eq!(got_mean_axis1, [5.0, 6.0, 7.0, 8.0]);

    let empty_input = dev.upload::<f32>(&[]).unwrap();
    let empty_input_layout = Layout::c_contiguous([3, 0]).unwrap();
    let empty_output = dev.upload(&[7.0f32; 3]).unwrap();
    let empty_output_layout = Layout::c_contiguous([3, 1]).unwrap();
    let prepared_empty_sum = prepare_sum_axis_into(
        &dev,
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
    .unwrap();
    prepared_empty_sum.dispatch(&dev).unwrap();
    let mut got_empty = [7.0f32; 3];
    dev.download(&empty_output, &mut got_empty).unwrap();
    assert_eq!(got_empty, [0.0, 0.0, 0.0]);

    let empty_product = prod_axis::<f32>(
        &dev,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        width,
    )
    .unwrap();
    let mut got_empty_product = [0.0f32; 3];
    dev.download(&empty_product, &mut got_empty_product)
        .unwrap();
    assert_eq!(got_empty_product, [1.0, 1.0, 1.0]);

    let empty_min = prepare_min_axis_into(
        &dev,
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
        empty_min,
        Err(HephaestusError::DispatchFailed { message })
            if message == "min_axis is undefined for empty axis 1"
    ));

    let alias_layout = Layout::c_contiguous([3, 1]).unwrap();
    let alias = prepare_sum_axis_into(
        &dev,
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

    let invalid_width = BlockWidth::new(3).unwrap();
    let invalid = prepare_sum_axis_into(
        &dev,
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
