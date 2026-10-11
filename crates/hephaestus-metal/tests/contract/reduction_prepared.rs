use super::*;

#[test]
fn prepared_reduction_reuses_device_outputs_and_batches() {
    let Some(d) = device("prepared_reduction_reuses_device_outputs_and_batches") else {
        return;
    };
    let input = d.upload(&[3.0f32, -2.0, 7.0, 1.0, 4.0]).unwrap();
    let width = BlockWidth::new(4).unwrap();

    let sum = prepare_reduction_with_width::<SumOp, _>(&d, &input, width).unwrap();
    sum.dispatch(&d).unwrap();
    let sum_output = sum.output();
    let mut got_sum = [0.0f32];
    d.download(&sum_output, &mut got_sum).unwrap();
    assert_eq!(got_sum, [13.0]);
    sum.dispatch(&d).unwrap();
    d.download(&sum.output(), &mut got_sum).unwrap();
    assert_eq!(got_sum, [13.0]);

    let min = prepare_reduction::<MinOp, _>(&d, &input).unwrap();
    let max = prepare_reduction::<MaxOp, _>(&d, &input).unwrap();
    submit_prepared_reduction_batch(&d, &[&min, &max]).unwrap();
    let mut got_min = [0.0f32];
    let mut got_max = [0.0f32];
    d.download(&min.output(), &mut got_min).unwrap();
    d.download(&max.output(), &mut got_max).unwrap();
    assert_eq!(got_min, [-2.0]);
    assert_eq!(got_max, [7.0]);

    let empty = d.upload::<f32>(&[]).unwrap();
    let prepared_empty = prepare_reduction::<SumOp, _>(&d, &empty).unwrap();
    prepared_empty.dispatch(&d).unwrap();
    let mut got_empty = [f32::NAN];
    d.download(&prepared_empty.output(), &mut got_empty)
        .unwrap();
    assert_eq!(got_empty, [0.0]);

    let invalid_width = BlockWidth::new(3).unwrap();
    assert!(matches!(
        prepare_reduction_with_width::<SumOp, _>(&d, &input, invalid_width),
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 3 must be a power of two"
    ));
}

#[test]
fn prepared_map_reductions_reuse_resources_and_validate_layouts() {
    let Some(device) = device("prepared_map_reductions_reuse_resources_and_validate_layouts")
    else {
        return;
    };

    let lhs = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let rhs = device.upload(&[5.0_f32, 6.0, 7.0, 8.0]).unwrap();
    let contiguous = Layout::c_contiguous([4]).unwrap();
    let prepared_dot = prepare_dot(
        &device,
        StridedOperand {
            buffer: &lhs,
            layout: &contiguous,
        },
        StridedOperand {
            buffer: &rhs,
            layout: &contiguous,
        },
    )
    .unwrap();
    let dot_output = prepared_dot.output().wgpu_buffer().raw().clone();
    prepared_dot.dispatch(&device).unwrap();
    let mut got = [0.0_f32];
    let dot_buffer = prepared_dot.output();
    device.download(&dot_buffer, &mut got).unwrap();
    assert_eq!(got, [70.0]);
    assert_eq!(&dot_output, dot_buffer.wgpu_buffer().raw());

    device
        .write_buffer(&lhs, &[2.0_f32, 2.0, 2.0, 2.0])
        .unwrap();
    prepared_dot.dispatch(&device).unwrap();
    let dot_buffer = prepared_dot.output();
    device.download(&dot_buffer, &mut got).unwrap();
    assert_eq!(got, [52.0]);
    assert_eq!(&dot_output, dot_buffer.wgpu_buffer().raw());

    let reversed = Layout::try_new([4], [-1], 3).expect("valid test layout");
    let reversed_dot = prepare_dot(
        &device,
        StridedOperand {
            buffer: &rhs,
            layout: &reversed,
        },
        StridedOperand {
            buffer: &lhs,
            layout: &contiguous,
        },
    )
    .unwrap();
    reversed_dot.dispatch(&device).unwrap();
    let reversed_buffer = reversed_dot.output();
    device.download(&reversed_buffer, &mut got).unwrap();
    assert_eq!(got, [52.0]);

    let norm_input = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let transposed = Layout::try_new([2, 2], [1, 2], 0).expect("valid test layout");
    let prepared_norm = prepare_norm_l2(
        &device,
        StridedOperand {
            buffer: &norm_input,
            layout: &transposed,
        },
    )
    .unwrap();
    let norm_output = prepared_norm.output().wgpu_buffer().raw().clone();
    prepared_norm.dispatch(&device).unwrap();
    let norm_buffer = prepared_norm.output();
    device.download(&norm_buffer, &mut got).unwrap();
    let expected = 30.0_f32.sqrt();
    assert!((got[0] - expected).abs() <= 2.0 * f32::EPSILON * expected.max(1.0));
    assert_eq!(&norm_output, norm_buffer.wgpu_buffer().raw());

    device.write_buffer(&norm_input, &[4.0_f32; 4]).unwrap();
    prepared_norm.dispatch(&device).unwrap();
    let norm_buffer = prepared_norm.output();
    device.download(&norm_buffer, &mut got).unwrap();
    assert_eq!(got, [8.0]);
    assert_eq!(&norm_output, norm_buffer.wgpu_buffer().raw());

    let empty = device.upload::<f32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0]).unwrap();
    let empty_dot = prepare_dot(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .unwrap();
    empty_dot.dispatch(&device).unwrap();
    let empty_dot_buffer = empty_dot.output();
    device.download(&empty_dot_buffer, &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let empty_norm = prepare_norm_l2(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .unwrap();
    empty_norm.dispatch(&device).unwrap();
    let empty_norm_buffer = empty_norm.output();
    device.download(&empty_norm_buffer, &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let wrong_shape = Layout::c_contiguous([3]).unwrap();
    assert!(matches!(
        prepare_dot(
            &device,
            StridedOperand {
                buffer: &lhs,
                layout: &contiguous,
            },
            StridedOperand {
                buffer: &rhs,
                layout: &wrong_shape,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("dot product shape mismatch:")
    ));

    let invalid_layout = Layout::try_new([3], [1], 2).expect("valid test layout");
    assert!(matches!(
        prepare_norm_l2(
            &device,
            StridedOperand {
                buffer: &lhs,
                layout: &invalid_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}
