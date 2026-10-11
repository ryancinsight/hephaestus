use super::*;

#[test]
fn reduction_sum_matches_cpu_reference() {
    let Some(d) = device("reduction_sum_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let out = reduction::<SumOp, f32>(&d, &a).unwrap();
    let mut host_out = [0.0f32; 1];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out[0], 10.0);
}

#[test]
fn reduction_axis_into_matches_cpu_reference() {
    let Some(d) = device("reduction_axis_into_matches_cpu_reference") else {
        return;
    };

    let input = d.upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let input_layout = Layout::c_contiguous([2, 3]).unwrap();
    let output = d.alloc_zeroed::<f32>(3).unwrap();
    let output_layout = Layout::c_contiguous([1, 3]).unwrap();

    reduce_axis_into::<SumOp, f32>(
        &d,
        StridedOperand {
            buffer: &input,
            layout: &input_layout,
        },
        0,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = [0.0f32; 3];
    d.download(&output, &mut got).unwrap();
    assert_eq!(got, [5.0, 7.0, 9.0]);

    let product_output = d.alloc_zeroed::<f32>(3).unwrap();
    let product_output_layout = Layout::c_contiguous([1, 3]).unwrap();
    reduce_axis_into::<ProdOp, f32>(
        &d,
        StridedOperand {
            buffer: &input,
            layout: &input_layout,
        },
        0,
        StridedOperand {
            buffer: &product_output,
            layout: &product_output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product_output = [0.0f32; 3];
    d.download(&product_output, &mut got_product_output)
        .unwrap();
    assert_eq!(got_product_output, [4.0, 10.0, 18.0]);

    let product = prod_axis(
        &d,
        StridedOperand {
            buffer: &input,
            layout: &input_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product = [0.0f32; 2];
    d.download(&product, &mut got_product).unwrap();
    assert_eq!(got_product, [6.0, 120.0]);

    let transposed_layout = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
    let transposed_product = prod_axis(
        &d,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_transposed_product = [0.0f32; 3];
    d.download(&transposed_product, &mut got_transposed_product)
        .unwrap();
    assert_eq!(got_transposed_product, [4.0, 10.0, 18.0]);

    let wrong_product_output = d.alloc_zeroed::<f32>(6).unwrap();
    let wrong_product_layout = Layout::c_contiguous([2, 3]).unwrap();
    assert!(matches!(
        prod_axis_into(
            &d,
            StridedOperand {
                buffer: &input,
                layout: &input_layout,
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
}
