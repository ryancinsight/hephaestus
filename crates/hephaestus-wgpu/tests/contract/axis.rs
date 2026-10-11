use super::*;

pub(crate) fn axis_reductions_match_leto_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = device.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([2, 3]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let leto_input = leto::Array::from_shape_vec([2, 3], host).unwrap();

    let out_axis0 = device.alloc_zeroed::<f32>(3).unwrap();
    let out_axis0_layout = Layout::c_contiguous([1, 3]).unwrap();
    sum_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &out_axis0,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_axis0 = leto_ops::sum_axis(&leto_input.view(), 0)
        .unwrap()
        .into_vec();
    let mut got_axis0 = vec![0.0f32; 3];
    device.download(&out_axis0, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, expected_axis0);

    let allocated_axis0 = sum_axis(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_axis0 = vec![0.0f32; 3];
    device
        .download(&allocated_axis0, &mut got_allocated_axis0)
        .unwrap();
    assert_eq!(got_allocated_axis0, expected_axis0);

    let product_axis0 =
        reduce_axis::<ProdOp, f32>(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let expected_product_axis0 = [4.0, 10.0, 18.0];
    let mut got_product_axis0 = vec![0.0f32; 3];
    device
        .download(&product_axis0, &mut got_product_axis0)
        .unwrap();
    assert_eq!(got_product_axis0, expected_product_axis0);

    let out_axis1 = device.alloc_zeroed::<f32>(2).unwrap();
    let out_axis1_layout = Layout::c_contiguous([2, 1]).unwrap();
    sum_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &out_axis1,
            layout: &out_axis1_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_axis1 = leto_ops::sum_axis(&leto_input.view(), 1)
        .unwrap()
        .into_vec();
    let mut got_axis1 = vec![0.0f32; 2];
    device.download(&out_axis1, &mut got_axis1).unwrap();
    assert_eq!(got_axis1, expected_axis1);

    let product_axis1 = prod_axis(&device, input_operand, 1, BlockWidth::DEFAULT).unwrap();
    let mut got_product_axis1 = vec![0.0f32; 2];
    device
        .download(&product_axis1, &mut got_product_axis1)
        .unwrap();
    assert_eq!(got_product_axis1, [6.0, 120.0]);

    let product_axis1_out = device.alloc_zeroed::<f32>(2).unwrap();
    let product_axis1_out_layout = Layout::c_contiguous([2, 1]).unwrap();
    prod_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &product_axis1_out,
            layout: &product_axis1_out_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product_axis1_into = [0.0f32; 2];
    device
        .download(&product_axis1_out, &mut got_product_axis1_into)
        .unwrap();
    assert_eq!(got_product_axis1_into, [6.0, 120.0]);

    let transposed_layout = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
    let transposed = StridedOperand {
        buffer: &input,
        layout: &transposed_layout,
    };
    let expected_transposed_product = [4.0, 10.0, 18.0];
    let transposed_product = prod_axis(&device, transposed, 1, BlockWidth::DEFAULT).unwrap();
    let mut got_transposed_product = vec![0.0f32; 3];
    device
        .download(&transposed_product, &mut got_transposed_product)
        .unwrap();
    assert_eq!(got_transposed_product, expected_transposed_product);

    let wrong_product_output = device.alloc_zeroed::<f32>(6).unwrap();
    let wrong_product_layout = Layout::c_contiguous([2, 3]).unwrap();
    assert!(matches!(
        prod_axis_into(
            &device,
            input_operand,
            1,
            StridedOperand {
                buffer: &wrong_product_output,
                layout: &wrong_product_layout,
            },
            BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("axis reduction output shape mismatch")
    ));

    let narrow_axis1 = sum_axis(&device, input_operand, 1, BlockWidth::new(2).unwrap()).unwrap();
    let mut got_narrow_axis1 = vec![0.0f32; 2];
    device
        .download(&narrow_axis1, &mut got_narrow_axis1)
        .unwrap();
    assert_eq!(got_narrow_axis1, expected_axis1);

    let min_axis0 = device.alloc_zeroed::<f32>(3).unwrap();
    min_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &min_axis0,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_min_axis0 = leto_ops::min_axis(&leto_input.view(), 0)
        .unwrap()
        .into_vec();
    let mut got_min_axis0 = vec![0.0f32; 3];
    device.download(&min_axis0, &mut got_min_axis0).unwrap();
    assert_eq!(got_min_axis0, expected_min_axis0);

    let allocated_min_axis0 = min_axis(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_min_axis0 = vec![0.0f32; 3];
    device
        .download(&allocated_min_axis0, &mut got_allocated_min_axis0)
        .unwrap();
    assert_eq!(got_allocated_min_axis0, expected_min_axis0);

    let max_axis1 = device.alloc_zeroed::<f32>(2).unwrap();
    max_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &max_axis1,
            layout: &out_axis1_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_max_axis1 = leto_ops::max_axis(&leto_input.view(), 1)
        .unwrap()
        .into_vec();
    let mut got_max_axis1 = vec![0.0f32; 2];
    device.download(&max_axis1, &mut got_max_axis1).unwrap();
    assert_eq!(got_max_axis1, expected_max_axis1);

    let allocated_max_axis1 = max_axis(&device, input_operand, 1, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_max_axis1 = vec![0.0f32; 2];
    device
        .download(&allocated_max_axis1, &mut got_allocated_max_axis1)
        .unwrap();
    assert_eq!(got_allocated_max_axis1, expected_max_axis1);

    let mean_axis0 = device.alloc_zeroed::<f32>(3).unwrap();
    mean_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &mean_axis0,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let expected_mean_axis0 = leto_ops::mean_axis(&leto_input.view(), 0)
        .unwrap()
        .into_vec();
    let mut got_mean_axis0 = vec![0.0f32; 3];
    device.download(&mean_axis0, &mut got_mean_axis0).unwrap();
    assert_eq!(got_mean_axis0, expected_mean_axis0);

    let allocated_mean_axis0 = mean_axis(&device, input_operand, 0, BlockWidth::DEFAULT).unwrap();
    let mut got_allocated_mean_axis0 = vec![0.0f32; 3];
    device
        .download(&allocated_mean_axis0, &mut got_allocated_mean_axis0)
        .unwrap();
    assert_eq!(got_allocated_mean_axis0, expected_mean_axis0);

    let prepared_sum_axis0_out = device.alloc_zeroed::<f32>(3).unwrap();
    let prepared_sum_axis0 = prepare_sum_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &prepared_sum_axis0_out,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    prepared_sum_axis0.dispatch(&device).unwrap();
    let mut got_prepared_sum_axis0 = vec![0.0f32; 3];
    device
        .download(&prepared_sum_axis0_out, &mut got_prepared_sum_axis0)
        .unwrap();
    assert_eq!(got_prepared_sum_axis0, expected_axis0);

    let batch_sum_axis0_out_a = device.alloc_zeroed::<f32>(3).unwrap();
    let batch_min_axis0_out = device.alloc_zeroed::<f32>(3).unwrap();
    let batch_sum_axis0_a = prepare_sum_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &batch_sum_axis0_out_a,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let batch_min_axis0 = prepare_min_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &batch_min_axis0_out,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    submit_prepared_axis_reduction_batch(&device, &[&batch_sum_axis0_a, &batch_min_axis0]).unwrap();
    let mut got_batch_sum_axis0_a = vec![0.0f32; 3];
    let mut got_batch_min_axis0 = vec![0.0f32; 3];
    device
        .download(&batch_sum_axis0_out_a, &mut got_batch_sum_axis0_a)
        .unwrap();
    device
        .download(&batch_min_axis0_out, &mut got_batch_min_axis0)
        .unwrap();
    assert_eq!(got_batch_sum_axis0_a, expected_axis0);
    assert_eq!(got_batch_min_axis0, expected_min_axis0);

    let prepared_min_axis0_out = device.alloc_zeroed::<f32>(3).unwrap();
    let prepared_min_axis0 = prepare_min_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &prepared_min_axis0_out,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    prepared_min_axis0.dispatch(&device).unwrap();
    let mut got_prepared_min_axis0 = vec![0.0f32; 3];
    device
        .download(&prepared_min_axis0_out, &mut got_prepared_min_axis0)
        .unwrap();
    assert_eq!(got_prepared_min_axis0, expected_min_axis0);

    let prepared_max_axis1_out = device.alloc_zeroed::<f32>(2).unwrap();
    let prepared_max_axis1 = prepare_max_axis_into(
        &device,
        input_operand,
        1,
        StridedOperand {
            buffer: &prepared_max_axis1_out,
            layout: &out_axis1_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    prepared_max_axis1.dispatch(&device).unwrap();
    let mut got_prepared_max_axis1 = vec![0.0f32; 2];
    device
        .download(&prepared_max_axis1_out, &mut got_prepared_max_axis1)
        .unwrap();
    assert_eq!(got_prepared_max_axis1, expected_max_axis1);

    let prepared_mean_axis0_out = device.alloc_zeroed::<f32>(3).unwrap();
    let prepared_mean_axis0 = prepare_mean_axis_into(
        &device,
        input_operand,
        0,
        StridedOperand {
            buffer: &prepared_mean_axis0_out,
            layout: &out_axis0_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    prepared_mean_axis0.dispatch(&device).unwrap();
    let mut got_prepared_mean_axis0 = vec![0.0f32; 3];
    device
        .download(&prepared_mean_axis0_out, &mut got_prepared_mean_axis0)
        .unwrap();
    assert_eq!(got_prepared_mean_axis0, expected_mean_axis0);

    let empty_input = device.upload::<f32>(&[]).unwrap();
    let empty_input_layout = Layout::c_contiguous([2, 0]).unwrap();
    let empty_output = device.upload(&[7.0f32; 2]).unwrap();
    let empty_output_layout = Layout::c_contiguous([2, 1]).unwrap();
    sum_axis_into(
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
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_empty = [7.0f32; 2];
    device.download(&empty_output, &mut got_empty).unwrap();
    assert_eq!(got_empty, [0.0, 0.0]);

    let empty_product = prod_axis(
        &device,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_empty_product = [0.0f32; 2];
    device
        .download(&empty_product, &mut got_empty_product)
        .unwrap();
    assert_eq!(got_empty_product, [1.0, 1.0]);

    let empty_axis0_input_layout = Layout::c_contiguous([0, 2]).unwrap();
    let empty_axis0_product = prod_axis(
        &device,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_axis0_input_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_empty_axis0_product = [0.0f32; 2];
    device
        .download(&empty_axis0_product, &mut got_empty_axis0_product)
        .unwrap();
    assert_eq!(got_empty_axis0_product, [1.0, 1.0]);

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
        BlockWidth::DEFAULT,
    )
    .unwrap();
    prepared_empty_sum.dispatch(&device).unwrap();
    device.download(&empty_output, &mut got_empty).unwrap();
    assert_eq!(got_empty, [0.0, 0.0]);

    let no_output_layout = Layout::c_contiguous([0, 1]).unwrap();
    let no_output = device.alloc_zeroed::<f32>(0).unwrap();
    let no_output_prepared = prepare_sum_axis_into(
        &device,
        StridedOperand {
            buffer: &empty_input,
            layout: &Layout::c_contiguous([0, 3]).unwrap(),
        },
        1,
        StridedOperand {
            buffer: &no_output,
            layout: &no_output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    submit_prepared_axis_reduction_batch::<f32>(&device, &[]).unwrap();
    submit_prepared_axis_reduction_batch(&device, &[&no_output_prepared]).unwrap();
    assert_eq!(no_output.len(), 0);
}
