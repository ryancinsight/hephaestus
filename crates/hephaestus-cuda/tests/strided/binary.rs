use super::common::{assert_dispatch_message, cpu_reference, device, dyn_op, op};
use hephaestus_core::{BlockWidth, ComputeDevice};
use hephaestus_cuda::{
    AddOp, MulOp, binary_elementwise_strided_dyn_into, binary_elementwise_strided_into,
};
use leto::Layout;

#[test]
fn strided_add_transposed_input_matches_cpu() {
    let Some(dev) = device("strided_add_transposed_input_matches_cpu") else {
        return;
    };
    let a_host: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a_layout = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
    let b_host: Vec<f32> = (0..6).map(|i| 10.0 * i as f32).collect();
    let b_layout = Layout::c_contiguous([2, 3]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let mut expected = vec![0.0f32; 6];
    cpu_reference(
        &a_host,
        &a_layout,
        &b_host,
        &b_layout,
        &mut expected,
        &out_layout,
        |x, y| x + y,
    );

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn strided_broadcast_inputs_match_cpu() {
    let Some(dev) = device("strided_broadcast_inputs_match_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 2.0];
    let a_layout = Layout::c_contiguous([2, 1]).unwrap();
    let b_host = vec![10.0f32, 20.0, 30.0];
    let b_layout = Layout::c_contiguous([1, 3]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let mut expected = vec![0.0f32; 6];
    cpu_reference(
        &a_host,
        &a_layout,
        &b_host,
        &b_layout,
        &mut expected,
        &out_layout,
        |x, y| x + y,
    );

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
    assert_eq!(got, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
}

#[test]
fn dynamic_strided_broadcast_inputs_match_cpu() {
    let Some(dev) = device("dynamic_strided_broadcast_inputs_match_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 2.0];
    let b_host = vec![10.0f32, 20.0, 30.0];

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();

    binary_elementwise_strided_dyn_into::<AddOp, f32>(
        &dev,
        dyn_op(&a, &[2, 1], &[1, 1], 0),
        dyn_op(&b, &[1, 3], &[3, 1], 0),
        dyn_op(&out, &[2, 3], &[3, 1], 0),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
}

#[test]
fn strided_offset_output_writes_only_selected_region() {
    let Some(dev) = device("strided_offset_output_writes_only_selected_region") else {
        return;
    };
    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let a_layout = Layout::c_contiguous([2, 2]).unwrap();
    let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
    let b_layout = Layout::c_contiguous([2, 2]).unwrap();
    let out_layout = Layout::try_new([2, 2], [3, 1], 4).expect("valid test layout");

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(9).unwrap();
    binary_elementwise_strided_into::<MulOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 9];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 0.0, 21.0, 32.0]);
}

#[test]
fn strided_rejects_aliasing_output_and_short_buffers() {
    let Some(dev) = device("strided_rejects_aliasing_output_and_short_buffers") else {
        return;
    };
    let a = dev.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let b = dev.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let out = dev.alloc_zeroed::<f32>(4).unwrap();

    let aliasing = Layout::try_new([2, 2], [0, 1], 0).expect("valid test layout");
    let flat = Layout::c_contiguous([2, 2]).unwrap();
    assert_dispatch_message(
        binary_elementwise_strided_into::<AddOp, f32, 2>(
            &dev,
            op(&a, &flat),
            op(&b, &flat),
            op(&out, &aliasing),
            BlockWidth::DEFAULT,
        ),
        "output layout must not contain zero-stride aliasing",
    );
}

#[test]
fn strided_rank3_batched_matches_cpu() {
    let Some(dev) = device("strided_rank3_batched_matches_cpu") else {
        return;
    };
    let a_host: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let a_layout = Layout::try_new([2, 3, 4], [12, 1, 3], 0).expect("valid test layout");
    let b_host: Vec<f32> = (0..24).map(|i| 100.0 - i as f32).collect();
    let b_layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3, 4]).unwrap();

    let mut expected = vec![0.0f32; 24];
    cpu_reference(
        &a_host,
        &a_layout,
        &b_host,
        &b_layout,
        &mut expected,
        &out_layout,
        |x, y| x * y,
    );

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(24).unwrap();
    binary_elementwise_strided_into::<MulOp, f32, 3>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 24];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn strided_rank8_batched_matches_cpu() {
    let Some(dev) = device("strided_rank8_batched_matches_cpu") else {
        return;
    };
    let shape = [2usize; 8];
    let a_strides = [1isize, 2, 4, 8, 16, 32, 64, 128];
    let a_dynamic_strides = [1usize, 2, 4, 8, 16, 32, 64, 128];
    let a_host: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    let a_layout = Layout::try_new(shape, a_strides, 0).expect("valid rank-eight layout");
    let b_host: Vec<f32> = (0..256).map(|i| 100.0 - i as f32).collect();
    let b_layout = Layout::c_contiguous(shape).unwrap();
    let out_layout = Layout::c_contiguous(shape).unwrap();

    let mut expected = vec![0.0f32; 256];
    cpu_reference(
        &a_host,
        &a_layout,
        &b_host,
        &b_layout,
        &mut expected,
        &out_layout,
        |x, y| x + y,
    );

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(256).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 8>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 256];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);

    let dynamic_out = dev.alloc_zeroed::<f32>(256).unwrap();
    binary_elementwise_strided_dyn_into::<AddOp, f32>(
        &dev,
        dyn_op(&a, &shape, &a_dynamic_strides, 0),
        dyn_op(&b, &shape, &[128, 64, 32, 16, 8, 4, 2, 1], 0),
        dyn_op(&dynamic_out, &shape, &[128, 64, 32, 16, 8, 4, 2, 1], 0),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut dynamic_got = vec![0.0f32; 256];
    dev.download(&dynamic_out, &mut dynamic_got).unwrap();
    assert_eq!(dynamic_got, expected);
}
