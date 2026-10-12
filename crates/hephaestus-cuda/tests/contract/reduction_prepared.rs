use super::*;

#[test]
fn prepared_reduction_reuses_device_outputs_and_batches() {
    let Some(dev) = device("prepared_reduction_reuses_device_outputs_and_batches") else {
        return;
    };

    let host: Vec<u32> = (0..1027).map(|index| (index % 17) as u32).collect();
    let expected_sum: u32 = host.iter().sum();
    let expected_min = host.iter().copied().min().expect("non-empty input");
    let expected_max = host.iter().copied().max().expect("non-empty input");
    let input = dev.upload(&host).unwrap();
    let width = BlockWidth::new(128).unwrap();

    let sum = prepare_reduction_with_width::<SumOp, _>(&dev, &input, width).unwrap();
    let output_ptr = sum.output().raw();
    let mut got = [u32::MAX];
    dev.download(sum.output(), &mut got).unwrap();
    assert_eq!(got, [0]);
    sum.dispatch().unwrap();
    dev.download(sum.output(), &mut got).unwrap();
    assert_eq!(got, [expected_sum]);
    sum.dispatch().unwrap();
    dev.download(sum.output(), &mut got).unwrap();
    assert_eq!(got, [expected_sum]);
    assert_eq!(sum.output().raw(), output_ptr);

    let min = prepare_reduction::<MinOp, _>(&dev, &input).unwrap();
    let max = prepare_reduction::<MaxOp, _>(&dev, &input).unwrap();
    submit_prepared_reduction_batch(&[&min, &max]).unwrap();
    let mut got_min = [0_u32];
    let mut got_max = [0_u32];
    dev.download(min.output(), &mut got_min).unwrap();
    dev.download(max.output(), &mut got_max).unwrap();
    assert_eq!(got_min, [expected_min]);
    assert_eq!(got_max, [expected_max]);

    let empty = dev.upload::<u32>(&[]).unwrap();
    let prepared_empty = prepare_reduction::<SumOp, _>(&dev, &empty).unwrap();
    prepared_empty.dispatch().unwrap();
    let mut got_empty = [u32::MAX];
    dev.download(prepared_empty.output(), &mut got_empty)
        .unwrap();
    assert_eq!(got_empty, [0]);

    let invalid_width = BlockWidth::new(192).unwrap();
    assert_dispatch_message(
        prepare_reduction_with_width::<SumOp, _>(&dev, &input, invalid_width),
        "reduction block width 192 must be a power of two",
    );
}

#[test]
fn prepared_map_reductions_reuse_resources_and_validate_layouts() {
    let Some(dev) = device("prepared_map_reductions_reuse_resources_and_validate_layouts") else {
        return;
    };

    let lhs = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let rhs = dev.upload(&[5.0_f32, 6.0, 7.0, 8.0]).unwrap();
    let contiguous = Layout::c_contiguous([4]).unwrap();
    let prepared_dot = prepare_dot(
        &dev,
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
    let dot_output = prepared_dot.output() as *const _;
    prepared_dot.dispatch().unwrap();
    let mut got = [0.0_f32];
    dev.download(prepared_dot.output(), &mut got).unwrap();
    assert_eq!(got, [70.0]);
    assert_eq!(dot_output, prepared_dot.output() as *const _);

    dev.write_buffer(&lhs, &[2.0_f32, 2.0, 2.0, 2.0]).unwrap();
    prepared_dot.dispatch().unwrap();
    dev.download(prepared_dot.output(), &mut got).unwrap();
    assert_eq!(got, [52.0]);
    assert_eq!(dot_output, prepared_dot.output() as *const _);

    let reversed = Layout::try_new([4], [-1], 3).expect("valid test layout");
    let reversed_dot = prepare_dot(
        &dev,
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
    reversed_dot.dispatch().unwrap();
    dev.download(reversed_dot.output(), &mut got).unwrap();
    assert_eq!(got, [52.0]);

    let norm_input = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let transposed = Layout::try_new([2, 2], [1, 2], 0).expect("valid test layout");
    let prepared_norm = prepare_norm_l2(
        &dev,
        StridedOperand {
            buffer: &norm_input,
            layout: &transposed,
        },
    )
    .unwrap();
    let norm_output = prepared_norm.output() as *const _;
    dev.write_buffer(prepared_norm.output(), &[f32::NAN])
        .unwrap();
    prepared_norm.dispatch().unwrap();
    dev.download(prepared_norm.output(), &mut got).unwrap();
    let expected = 30.0_f32.sqrt();
    assert!((got[0] - expected).abs() <= 2.0 * f32::EPSILON * expected.max(1.0));
    assert_eq!(norm_output, prepared_norm.output() as *const _);

    dev.write_buffer(&norm_input, &[4.0_f32; 4]).unwrap();
    dev.write_buffer(prepared_norm.output(), &[f32::NAN])
        .unwrap();
    prepared_norm.dispatch().unwrap();
    dev.download(prepared_norm.output(), &mut got).unwrap();
    assert_eq!(got, [8.0]);
    assert_eq!(norm_output, prepared_norm.output() as *const _);

    let empty = dev.upload::<f32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0]).unwrap();
    let empty_dot = prepare_dot(
        &dev,
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
    empty_dot.dispatch().unwrap();
    dev.download(empty_dot.output(), &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let empty_norm = prepare_norm_l2(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .unwrap();
    empty_norm.dispatch().unwrap();
    dev.download(empty_norm.output(), &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let wrong_shape = Layout::c_contiguous([3]).unwrap();
    assert!(matches!(
        prepare_dot(
            &dev,
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
            &dev,
            StridedOperand {
                buffer: &lhs,
                layout: &invalid_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}
