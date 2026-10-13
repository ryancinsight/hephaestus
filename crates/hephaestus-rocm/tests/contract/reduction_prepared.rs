use super::*;

#[test]
fn prepared_reduction_reuses_device_outputs_and_batches() {
    let Some(device) = device("prepared_reduction_reuses_device_outputs_and_batches") else {
        return;
    };

    let input: Vec<u32> = (0..1027_u32).map(|index| index % 17).collect();
    let expected_sum: u32 = input.iter().sum();
    let expected_min = input.iter().copied().min().expect("non-empty input");
    let expected_max = input.iter().copied().max().expect("non-empty input");
    let input_buffer = device.upload(&input).expect("HIP prepared input upload");
    let width = BlockWidth::new(128).expect("prepared reduction width");

    let sum = prepare_reduction_with_width::<SumOp, _>(&device, &input_buffer, width)
        .expect("HIP prepared sum");
    let mut got_sum = [u32::MAX];
    device
        .download(sum.output(), &mut got_sum)
        .expect("HIP initialized prepared sum download");
    assert_eq!(got_sum, [0]);
    sum.dispatch().expect("HIP prepared sum dispatch");
    device
        .download(sum.output(), &mut got_sum)
        .expect("HIP prepared sum download");
    assert_eq!(got_sum, [expected_sum]);
    sum.dispatch().expect("HIP prepared sum repeat dispatch");
    device
        .download(sum.output(), &mut got_sum)
        .expect("HIP prepared repeated sum download");
    assert_eq!(got_sum, [expected_sum]);

    let min = prepare_reduction::<MinOp, _>(&device, &input_buffer).expect("HIP prepared min");
    let max = prepare_reduction::<MaxOp, _>(&device, &input_buffer).expect("HIP prepared max");
    submit_prepared_reduction_batch(&[&min, &max]).expect("HIP prepared reduction batch");
    let mut got_min = [0_u32];
    let mut got_max = [0_u32];
    device
        .download(min.output(), &mut got_min)
        .expect("HIP prepared min download");
    device
        .download(max.output(), &mut got_max)
        .expect("HIP prepared max download");
    assert_eq!(got_min, [expected_min]);
    assert_eq!(got_max, [expected_max]);

    let empty = device
        .upload::<u32>(&[])
        .expect("HIP prepared empty input upload");
    let prepared_empty =
        prepare_reduction::<SumOp, _>(&device, &empty).expect("HIP prepared empty sum");
    prepared_empty
        .dispatch()
        .expect("HIP prepared empty dispatch");
    let mut got_empty = [u32::MAX];
    device
        .download(prepared_empty.output(), &mut got_empty)
        .expect("HIP prepared empty download");
    assert_eq!(got_empty, [0]);

    let invalid_width = BlockWidth::new(192).expect("prepared invalid width");
    assert!(matches!(
        prepare_reduction_with_width::<SumOp, _>(&device, &input_buffer, invalid_width),
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 192 must be a power of two"
    ));
}

#[test]
fn prepared_map_reductions_reuse_resources_and_validate_layouts() {
    let Some(device) = device("prepared_map_reductions_reuse_resources_and_validate_layouts")
    else {
        return;
    };

    let lhs = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("HIP dot lhs upload");
    let rhs = device
        .upload(&[5.0_f32, 6.0, 7.0, 8.0])
        .expect("HIP dot rhs upload");
    let contiguous = Layout::c_contiguous([4]).expect("HIP dot layout");
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
    .expect("HIP prepared dot");
    let dot_output = std::ptr::from_ref(prepared_dot.output());
    prepared_dot.dispatch().expect("HIP prepared dot dispatch");
    let mut got = [0.0_f32];
    device
        .download(prepared_dot.output(), &mut got)
        .expect("HIP prepared dot download");
    assert_eq!(got, [70.0]);
    assert_eq!(dot_output, std::ptr::from_ref(prepared_dot.output()));

    device
        .write_buffer(&lhs, &[2.0_f32, 2.0, 2.0, 2.0])
        .expect("HIP prepared dot input update");
    prepared_dot
        .dispatch()
        .expect("HIP repeated prepared dot dispatch");
    device
        .download(prepared_dot.output(), &mut got)
        .expect("HIP repeated prepared dot download");
    assert_eq!(got, [52.0]);
    assert_eq!(dot_output, std::ptr::from_ref(prepared_dot.output()));

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
    .expect("HIP prepared reversed dot");
    reversed_dot
        .dispatch()
        .expect("HIP prepared reversed dot dispatch");
    device
        .download(reversed_dot.output(), &mut got)
        .expect("HIP prepared reversed dot download");
    assert_eq!(got, [52.0]);

    let norm_input = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("HIP norm input upload");
    let transposed = Layout::try_new([2, 2], [1, 2], 0).expect("valid test layout");
    let prepared_norm = prepare_norm_l2(
        &device,
        StridedOperand {
            buffer: &norm_input,
            layout: &transposed,
        },
    )
    .expect("HIP prepared L2 norm");
    let norm_output = std::ptr::from_ref(prepared_norm.output());
    device
        .write_buffer(prepared_norm.output(), &[f32::NAN])
        .expect("HIP prepared L2 output poison");
    prepared_norm
        .dispatch()
        .expect("HIP prepared L2 norm dispatch");
    device
        .download(prepared_norm.output(), &mut got)
        .expect("HIP prepared L2 norm download");
    let expected = 30.0_f32.sqrt();
    assert!((got[0] - expected).abs() <= 2.0 * f32::EPSILON * expected.max(1.0));
    assert_eq!(norm_output, std::ptr::from_ref(prepared_norm.output()));

    device
        .write_buffer(&norm_input, &[4.0_f32; 4])
        .expect("HIP prepared L2 input update");
    device
        .write_buffer(prepared_norm.output(), &[f32::NAN])
        .expect("HIP repeated prepared L2 output poison");
    prepared_norm
        .dispatch()
        .expect("HIP repeated prepared L2 norm dispatch");
    device
        .download(prepared_norm.output(), &mut got)
        .expect("HIP repeated prepared L2 norm download");
    assert_eq!(got, [8.0]);
    assert_eq!(norm_output, std::ptr::from_ref(prepared_norm.output()));

    let empty = device.upload::<f32>(&[]).expect("HIP empty map input");
    let empty_layout = Layout::c_contiguous([0]).expect("HIP empty map layout");
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
    .expect("HIP prepared empty dot");
    empty_dot
        .dispatch()
        .expect("HIP prepared empty dot dispatch");
    device
        .download(empty_dot.output(), &mut got)
        .expect("HIP prepared empty dot download");
    assert_eq!(got, [0.0]);

    let empty_norm = prepare_norm_l2(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .expect("HIP prepared empty norm");
    empty_norm
        .dispatch()
        .expect("HIP prepared empty norm dispatch");
    device
        .download(empty_norm.output(), &mut got)
        .expect("HIP prepared empty norm download");
    assert_eq!(got, [0.0]);

    let wrong_shape = Layout::c_contiguous([3]).expect("HIP wrong dot shape");
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
