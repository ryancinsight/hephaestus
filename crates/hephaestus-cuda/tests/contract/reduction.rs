use super::*;

#[test]
fn reduction_sum_matches_cpu_reference() {
    let Some(dev) = device("reduction_sum_matches_cpu_reference") else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
        // f32
        let host_f32: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let expected_f32: f32 = host_f32.iter().sum();
        let buf_f32 = dev.upload(&host_f32).unwrap();
        let out_f32 = reduction::<SumOp, f32>(&dev, &buf_f32).unwrap();
        let mut got_f32 = vec![0.0f32; 1];
        dev.download(&out_f32, &mut got_f32).unwrap();
        assert_eq!(
            got_f32[0], expected_f32,
            "f32 sum mismatch at size {}",
            size
        );

        // u32
        let host_u32: Vec<u32> = (0..size).map(|i| i as u32).collect();
        let expected_u32: u32 = host_u32.iter().sum();
        let buf_u32 = dev.upload(&host_u32).unwrap();
        let out_u32 = reduction::<SumOp, u32>(&dev, &buf_u32).unwrap();
        let mut got_u32 = vec![0u32; 1];
        dev.download(&out_u32, &mut got_u32).unwrap();
        assert_eq!(
            got_u32[0], expected_u32,
            "u32 sum mismatch at size {}",
            size
        );

        // i32
        let host_i32: Vec<i32> = (0..size).map(|i| if i % 2 == 0 { i } else { -i }).collect();
        let expected_i32: i32 = host_i32.iter().sum();
        let buf_i32 = dev.upload(&host_i32).unwrap();
        let out_i32 = reduction::<SumOp, i32>(&dev, &buf_i32).unwrap();
        let mut got_i32 = vec![0i32; 1];
        dev.download(&out_i32, &mut got_i32).unwrap();
        assert_eq!(
            got_i32[0], expected_i32,
            "i32 sum mismatch at size {}",
            size
        );
    }
}

#[test]
fn reduction_min_max_matches_cpu_reference() {
    let Some(dev) = device("reduction_min_max_matches_cpu_reference") else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
        let host_f32: Vec<f32> = (0..size)
            .map(|i| (i as f32 * 12.34 - 100.0).sin())
            .collect();
        let expected_min_f32 = if size == 0 {
            f32::MAX
        } else {
            host_f32.iter().copied().fold(f32::NAN, f32::min)
        };
        let expected_max_f32 = if size == 0 {
            f32::MIN
        } else {
            host_f32.iter().copied().fold(f32::NAN, f32::max)
        };

        let buf_f32 = dev.upload(&host_f32).unwrap();

        let out_min_f32 = reduction::<MinOp, f32>(&dev, &buf_f32).unwrap();
        let mut got_min_f32 = vec![0.0f32; 1];
        dev.download(&out_min_f32, &mut got_min_f32).unwrap();
        assert_eq!(
            got_min_f32[0], expected_min_f32,
            "f32 min mismatch at size {}",
            size
        );

        let out_max_f32 = reduction::<MaxOp, f32>(&dev, &buf_f32).unwrap();
        let mut got_max_f32 = vec![0.0f32; 1];
        dev.download(&out_max_f32, &mut got_max_f32).unwrap();
        assert_eq!(
            got_max_f32[0], expected_max_f32,
            "f32 max mismatch at size {}",
            size
        );
    }
}

#[test]
fn reduction_width_is_part_of_dispatch_contract() {
    let Some(dev) = device("reduction_width_is_part_of_dispatch_contract") else {
        return;
    };

    let host: Vec<u32> = (0..1027).collect();
    let expected: u32 = host.iter().sum();
    let input = dev.upload(&host).unwrap();

    let narrow = BlockWidth::new(128).unwrap();
    let out_narrow = reduction_with_width::<SumOp, u32>(&dev, &input, narrow).unwrap();
    let mut got_narrow = vec![0u32; 1];
    dev.download(&out_narrow, &mut got_narrow).unwrap();
    assert_eq!(got_narrow[0], expected);

    let non_power = BlockWidth::new(192).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&dev, &input, non_power),
        "reduction block width 192 must be a power of two",
    );
}

#[test]
fn linalg_norm_preserves_large_periodic_sum() {
    let Some(dev) = device("linalg_norm_preserves_large_periodic_sum") else {
        return;
    };
    let host: Vec<f32> = (0..1 << 20)
        .map(|index| (index % 17) as f32 - 8.0)
        .collect();
    let input = dev.upload(&host).unwrap();
    let layout = Layout::c_contiguous([host.len()]).unwrap();
    let operand = StridedOperand {
        buffer: &input,
        layout: &layout,
    };

    // 61_680 complete periods of [-8, ..., 8] contribute 408 each;
    // the remaining [-8, ..., 7] contributes 344. Every partial through
    // the penultimate binary reduction level is an integer below 2^24,
    // and the final sum 25_165_784 is even and exactly representable.
    // Both paths therefore reach the same correctly rounded square root.
    let expected = 25_165_784.0_f32.sqrt();
    let immediate = norm_l2(&dev, operand).unwrap();
    let prepared = prepare_norm_l2(&dev, operand).unwrap();
    prepared.dispatch().unwrap();

    for output in [&immediate, prepared.output()] {
        let mut actual = [0.0_f32; 1];
        dev.download(output, &mut actual).unwrap();
        assert_eq!(actual[0], expected);
    }
}

#[test]
fn reduction_axis_reduction_generic_matches_cpu() {
    let Some(dev) = device("reduction_axis_reduction_generic_matches_cpu") else {
        return;
    };

    let host_in = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = dev.upload(&host_in).unwrap();
    let a_layout = Layout::c_contiguous([2, 3]).unwrap();

    // Sum axis 0
    let out = reduce_axis::<SumOp, f32>(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = vec![0.0f32; 3];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![5.0, 7.0, 9.0]);

    let product_columns = reduce_axis::<ProdOp, f32>(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product_columns = [0.0f32; 3];
    dev.download(&product_columns, &mut got_product_columns)
        .unwrap();
    assert_eq!(got_product_columns, [4.0, 10.0, 18.0]);

    let product = prod_axis::<f32>(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product = vec![0.0f32; 2];
    dev.download(&product, &mut got_product).unwrap();
    assert_eq!(got_product, vec![6.0, 120.0]);

    let product_output = dev.alloc_zeroed::<f32>(2).unwrap();
    let product_output_layout = Layout::c_contiguous([2, 1]).unwrap();
    prod_axis_into(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &a_layout,
        },
        1,
        StridedOperand {
            buffer: &product_output,
            layout: &product_output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product_into = [0.0f32; 2];
    dev.download(&product_output, &mut got_product_into)
        .unwrap();
    assert_eq!(got_product_into, [6.0, 120.0]);

    let transposed_layout = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
    let transposed_product = prod_axis(
        &dev,
        StridedOperand {
            buffer: &a,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_transposed_product = [0.0f32; 3];
    dev.download(&transposed_product, &mut got_transposed_product)
        .unwrap();
    assert_eq!(got_transposed_product, [4.0, 10.0, 18.0]);

    let wrong_product_output = dev.alloc_zeroed::<f32>(6).unwrap();
    let wrong_product_layout = Layout::c_contiguous([2, 3]).unwrap();
    assert!(matches!(
        prod_axis_into(
            &dev,
            StridedOperand {
                buffer: &a,
                layout: &a_layout,
            },
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

    let empty = dev.upload::<f32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0, 2]).unwrap();
    let empty_product = prod_axis(
        &dev,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_empty_product = [0.0f32; 2];
    dev.download(&empty_product, &mut got_empty_product)
        .unwrap();
    assert_eq!(got_empty_product, [1.0, 1.0]);
}
