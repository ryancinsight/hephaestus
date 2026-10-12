use super::common::{device, op};
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer};
use hephaestus_cuda::{
    AddOp, EqOp, NegOp, binary_elementwise_strided, binary_elementwise_strided_into,
    binary_elementwise_strided_typed, scalar_elementwise_strided, scalar_elementwise_strided_into,
    unary_elementwise_strided,
};
use leto::Layout;

#[test]
fn strided_scalar_matches_binary_broadcast_semantics() {
    let Some(dev) = device("strided_scalar_matches_binary_broadcast_semantics") else {
        return;
    };
    let a_host = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a_layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    scalar_elementwise_strided_into::<AddOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        100.0,
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
}

#[test]
fn allocating_strided_forms_fully_overwrite_dense_outputs() {
    let Some(dev) = device("allocating_strided_forms_fully_overwrite_dense_outputs") else {
        return;
    };
    let input = dev.upload(&[1_i32, 4, 2, 5, 3, 6]).unwrap();
    let transposed = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let rhs = dev.upload(&[10_i32, 20, 30, 40, 50, 60]).unwrap();
    let dense = Layout::c_contiguous([2, 3]).unwrap();
    let probe = dev.upload(&[1_i32, 9, 3, 4, 0, 6]).unwrap();

    let binary = binary_elementwise_strided::<AddOp, _, 2>(
        &dev,
        op(&input, &transposed),
        op(&rhs, &dense),
        [2, 3],
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let typed = binary_elementwise_strided_typed::<EqOp, _, 2>(
        &dev,
        op(&input, &transposed),
        op(&probe, &dense),
        [2, 3],
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let unary = unary_elementwise_strided::<NegOp, _, 2>(
        &dev,
        op(&input, &transposed),
        [2, 3],
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let scalar = scalar_elementwise_strided::<AddOp, _, 2>(
        &dev,
        op(&input, &transposed),
        10,
        [2, 3],
        BlockWidth::DEFAULT,
    )
    .unwrap();

    for (buffer, expected) in [
        (&binary, [11, 22, 33, 44, 55, 66]),
        (&typed, [1, 0, 1, 1, 0, 1]),
        (&unary, [-1, -2, -3, -4, -5, -6]),
        (&scalar, [11, 12, 13, 14, 15, 16]),
    ] {
        let mut got = [0_i32; 6];
        dev.download(buffer, &mut got).unwrap();
        assert_eq!(got, expected);
    }

    let empty_input = dev.upload::<i32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0, 3]).unwrap();
    let empty = unary_elementwise_strided::<NegOp, _, 2>(
        &dev,
        op(&empty_input, &empty_layout),
        [0, 3],
        BlockWidth::DEFAULT,
    )
    .unwrap();
    assert_eq!(empty.len(), 0);
}

#[test]
fn non_default_block_width_produces_identical_results() {
    let Some(dev) = device("non_default_block_width_produces_identical_results") else {
        return;
    };
    let len = 1027usize;
    let a_host: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
    let b_host: Vec<f32> = (0..len).map(|i| 1000.0 - i as f32).collect();
    let layout = Layout::c_contiguous([len]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();

    let narrow = BlockWidth::new(128).unwrap();
    let out_narrow = dev.alloc_zeroed::<f32>(len).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 1>(
        &dev,
        op(&a, &layout),
        op(&b, &layout),
        op(&out_narrow, &layout),
        narrow,
    )
    .unwrap();

    let out_default = dev.alloc_zeroed::<f32>(len).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 1>(
        &dev,
        op(&a, &layout),
        op(&b, &layout),
        op(&out_default, &layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got_narrow = vec![0.0f32; len];
    let mut got_default = vec![0.0f32; len];
    dev.download(&out_narrow, &mut got_narrow).unwrap();
    dev.download(&out_default, &mut got_default).unwrap();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();
    assert_eq!(got_narrow, expected);
    assert_eq!(got_default, expected);
}
