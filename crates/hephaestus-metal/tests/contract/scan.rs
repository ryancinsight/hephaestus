use super::*;

#[test]
fn scan_cumprod_convenience_preserves_strided_and_empty_contract() {
    let Some(device) = device("scan_cumprod_convenience_preserves_strided_and_empty_contract")
    else {
        return;
    };

    let physical = vec![1_i32, 2, 3, 4, 5, 6];
    let input = device.upload(&physical).unwrap();
    let transposed_layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let output_layout = Layout::c_contiguous([2, 3]).unwrap();
    let output = device.alloc_zeroed::<i32>(6).unwrap();
    cumprod_into(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = [0_i32; 6];
    device.download(&output, &mut got).unwrap();
    assert_eq!(got, [1, 3, 15, 2, 8, 48]);

    let allocated = cumprod(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_allocated = [0_i32; 6];
    device.download(&allocated, &mut got_allocated).unwrap();
    assert_eq!(got_allocated, [1, 3, 5, 2, 12, 30]);

    let suffix_product = suffix_prod(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_product = [0_i32; 6];
    device
        .download(&suffix_product, &mut got_suffix_product)
        .unwrap();
    assert_eq!(got_suffix_product, [15, 15, 5, 48, 24, 6]);

    let suffix_product_into = device.alloc_zeroed::<i32>(6).unwrap();
    suffix_prod_into(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        StridedOperand {
            buffer: &suffix_product_into,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_product_into = [0_i32; 6];
    device
        .download(&suffix_product_into, &mut got_suffix_product_into)
        .unwrap();
    assert_eq!(got_suffix_product_into, [2, 12, 30, 2, 4, 6]);

    let suffix = suffix_sum(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix = [0_i32; 6];
    device.download(&suffix, &mut got_suffix).unwrap();
    assert_eq!(got_suffix, [9, 8, 5, 12, 10, 6]);

    let suffix_into = device.alloc_zeroed::<i32>(6).unwrap();
    suffix_sum_into(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        StridedOperand {
            buffer: &suffix_into,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_into = [0_i32; 6];
    device.download(&suffix_into, &mut got_suffix_into).unwrap();
    assert_eq!(got_suffix_into, [3, 7, 11, 2, 4, 6]);

    let empty = device.alloc_zeroed::<i32>(0).unwrap();
    let empty_layout = Layout::c_contiguous([2, 0]).unwrap();
    let empty_output = cumprod(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    assert_eq!(empty_output.len(), 0);

    let invalid_layout = Layout::try_new([2, 3], [1, 2], 1).expect("valid test layout");
    assert!(matches!(
        cumprod(
            &device,
            StridedOperand {
                buffer: &input,
                layout: &invalid_layout,
            },
            1,
            BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}
