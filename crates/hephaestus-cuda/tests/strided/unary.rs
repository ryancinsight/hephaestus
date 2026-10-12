use super::common::{device, dyn_op, op};
use hephaestus_core::{BlockWidth, ComputeDevice};
use hephaestus_cuda::{
    NegOp, SqrtOp, unary_elementwise_strided_dyn_into, unary_elementwise_strided_into,
};
use leto::Layout;

#[test]
fn strided_unary_transposed_matches_cpu() {
    let Some(dev) = device("strided_unary_transposed_matches_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 9.0, 25.0, 4.0, 16.0, 36.0];
    let a_layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    unary_elementwise_strided_into::<SqrtOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![1.0, 5.0, 4.0, 3.0, 2.0, 6.0]);
}

#[test]
fn strided_unary_broadcasts_input_to_output_shape() {
    let Some(dev) = device("strided_unary_broadcasts_input_to_output_shape") else {
        return;
    };
    let a_host = vec![1.0f32, -2.0, 3.0];
    let a_layout = Layout::c_contiguous([1, 3]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    unary_elementwise_strided_into::<NegOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![-1.0, 2.0, -3.0, -1.0, 2.0, -3.0]);
}

#[test]
fn dynamic_strided_unary_transposed_matches_cpu() {
    let Some(dev) = device("dynamic_strided_unary_transposed_matches_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 9.0, 25.0, 4.0, 16.0, 36.0];

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    unary_elementwise_strided_dyn_into::<SqrtOp, f32>(
        &dev,
        dyn_op(&a, &[2, 3], &[1, 2], 0),
        dyn_op(&out, &[2, 3], &[3, 1], 0),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![1.0, 5.0, 4.0, 3.0, 2.0, 6.0]);
}
