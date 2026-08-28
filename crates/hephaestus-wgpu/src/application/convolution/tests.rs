use hephaestus_core::{
    ComputeDevice, ConvolutionBackwardOperands, ConvolutionForwardOperands,
    ConvolutionGradientViews, ConvolutionOps, StridedView,
};
use leto::{
    ArrayView, ArrayViewMut, ConvolutionParameters, Layout, TransposedConvolutionParameters,
};
use leto_ops::{
    TransposedConvolutionGradients, convolution_backward_accumulate, convolution_forward_into,
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into,
};

use super::WgpuConvolutionOps;
use crate::WgpuDevice;

mod numerical;

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-convolution-test") {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping WGPU convolution test: {error}");
            None
        }
    }
}

#[test]
fn module_cases_share_process_state() {
    crate::test_support::run_cases(&[
        (
            "matches_leto_regular_and_transposed_forward_backward",
            matches_leto_regular_and_transposed_forward_backward as fn(),
        ),
        (
            "matches_leto_spatial_ranks_two_and_three",
            matches_leto_spatial_ranks_two_and_three as fn(),
        ),
        (
            "rejects_forward_buffer_alias_before_mutation",
            rejects_forward_buffer_alias_before_mutation as fn(),
        ),
        (
            "rejects_prepared_kernel_on_a_different_device",
            rejects_prepared_kernel_on_a_different_device as fn(),
        ),
    ]);
}

fn matches_leto_regular_and_transposed_forward_backward() {
    let Some(device) = device_or_skip() else {
        return;
    };
    verify_regular_case(
        &device,
        &[1.0, 2.0, 3.0, 4.0],
        Layout::try_new([1, 1, 4], [4, 4, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[2.0, -1.0],
        Layout::try_new([1, 1, 2], [2, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[1.0, -2.0, 3.0],
        Layout::try_new([1, 1, 3], [3, 3, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        ConvolutionParameters::new([1], [0], [1]).expect("valid regular parameters"),
    );
    verify_transposed_case(
        &device,
        &[1.0, -2.0, 3.0],
        Layout::try_new([1, 1, 3], [3, 3, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[0.5, 2.0],
        Layout::try_new([1, 1, 2], [2, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[1.0, 2.0, -1.0, 0.5, 3.0],
        Layout::try_new([1, 1, 5], [5, 5, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        TransposedConvolutionParameters::new([2], [1], [1], [1])
            .expect("valid transposed parameters"),
    );
}

fn matches_leto_spatial_ranks_two_and_three() {
    let Some(device) = device_or_skip() else {
        return;
    };
    verify_regular_case(
        &device,
        &[1.0, -2.0, 3.0, 0.5, 4.0, -1.0, 2.0, 1.5, -3.0],
        Layout::try_new([1, 1, 3, 3], [9, 9, 3, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[0.5, -1.0, 2.0, 0.25],
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[1.0, -2.0, 0.5, 3.0],
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        ConvolutionParameters::new([1, 1], [0, 0], [1, 1])
            .expect("valid rank-two regular parameters"),
    );
    verify_transposed_case(
        &device,
        &[1.0, -2.0, 0.5, 3.0],
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[0.5, -1.0, 2.0, 0.25],
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 0.25, -0.75, 4.0],
        Layout::try_new([1, 1, 3, 3], [9, 9, 3, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        TransposedConvolutionParameters::new([2, 1], [1, 0], [1, 0], [1, 1])
            .expect("valid rank-two transposed parameters"),
    );
    verify_regular_case(
        &device,
        &[
            1.0, -2.0, 3.0, 0.5, 4.0, -1.0, 2.0, 1.5, -3.0, 0.25, -0.5, 2.5, 1.25, -1.25, 0.75,
            3.5, -2.5, 1.75, -0.25, 2.25, -3.25, 4.25, -1.5, 0.5, 1.0, -2.0, 3.0,
        ],
        Layout::try_new([1, 1, 3, 3, 3], [27, 27, 9, 3, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.0, 0.125],
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 0.25, -0.75],
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        ConvolutionParameters::new([1, 1, 1], [0, 0, 0], [1, 1, 1])
            .expect("valid rank-three regular parameters"),
    );
    verify_transposed_case(
        &device,
        &[1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 0.25, -0.75],
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.0, 0.125],
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        &[
            1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 0.25, -0.75, 4.0, -0.5, 1.5, -2.5, 3.5, 0.75, -1.25,
            2.25, -3.25, 4.25, 0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0, 1.125,
        ],
        Layout::try_new([1, 1, 3, 3, 3], [27, 27, 9, 3, 1], 0)
            .expect("invariant: submatrix layout derives from a validated parent"),
        TransposedConvolutionParameters::new([2, 1, 2], [1, 0, 1], [1, 0, 1], [1, 1, 1])
            .expect("valid rank-three transposed parameters"),
    );
}

fn rejects_forward_buffer_alias_before_mutation() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let initial = [1.0_f32, -2.0, 3.0, 0.5, 4.0, -1.0, 2.0, 1.5, -3.0];
    let shared = device.upload(&initial).expect("shared buffer upload");
    let weight = device
        .upload(&[0.5_f32, -1.0, 2.0, 0.25])
        .expect("weight upload");
    let input_layout = Layout::try_new([1, 1, 3, 3], [9, 9, 3, 1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let weight_layout = Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let output_layout = Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(&shared, &input_layout),
        weight: StridedView::new(&weight, &weight_layout),
        bias: None,
        output: StridedView::new(&shared, &output_layout),
    };
    let parameters = ConvolutionParameters::new([1, 1], [0, 0], [1, 1])
        .expect("valid rank-two regular parameters");

    let error =
        match <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::prepare_convolution_forward(
            &WgpuConvolutionOps,
            &device,
            operands,
            parameters,
        ) {
            Ok(_) => panic!("aliased forward preparation must fail"),
            Err(error) => error,
        };

    assert_eq!(
        error.to_string(),
        "invalid configuration: convolution writable buffers must not alias readable operands or each other"
    );
    assert_download_eq(&device, &shared, &initial);
}

fn rejects_prepared_kernel_on_a_different_device() {
    let Some(source_device) = device_or_skip() else {
        return;
    };
    let Some(other_device) = device_or_skip() else {
        return;
    };
    let input_host = [1.0_f32, 2.0, 3.0];
    let input = source_device.upload(&input_host).expect("input upload");
    let weight = source_device
        .upload(&[0.5_f32, -1.0])
        .expect("weight upload");
    let output = source_device
        .alloc_zeroed::<f32>(2)
        .expect("output allocation");
    let input_layout = Layout::try_new([1, 1, 3], [3, 3, 1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let weight_layout = Layout::try_new([1, 1, 2], [2, 2, 1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let output_layout = Layout::try_new([1, 1, 2], [2, 2, 1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let parameters = ConvolutionParameters::new([1], [0], [1]).expect("valid regular parameters");
    let operations = WgpuConvolutionOps;
    let prepared = operations
        .prepare_convolution_forward(
            &source_device,
            ConvolutionForwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                bias: None,
                output: StridedView::new(&output, &output_layout),
            },
            parameters,
        )
        .expect("prepare regular forward");

    let error = <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::dispatch_convolution_forward::<3, 1>(
        &operations,
        &other_device,
        &prepared,
    )
    .expect_err("cross-device dispatch must fail");

    assert_eq!(
        error.to_string(),
        "kernel dispatch failed: prepared WGPU convolution belongs to a different device"
    );
    assert_download_eq(&source_device, &output, &[0.0, 0.0]);
}

#[expect(
    clippy::too_many_arguments,
    reason = "the test driver receives one complete convolution case"
)]
fn verify_regular_case<const R: usize, const S: usize>(
    device: &WgpuDevice,
    input_host: &[f32],
    input_layout: Layout<R>,
    weight_host: &[f32],
    weight_layout: Layout<R>,
    grad_output_host: &[f32],
    output_layout: Layout<R>,
    parameters: ConvolutionParameters<S>,
) {
    let bias_host = [0.5_f32];
    let bias_layout = Layout::try_new([1], [1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let mut expected_output = vec![0.0_f32; grad_output_host.len()];
    convolution_forward_into(
        &ArrayView::new(input_layout, input_host),
        &ArrayView::new(weight_layout, weight_host),
        Some(&ArrayView::new(bias_layout, &bias_host)),
        parameters,
        &mut ArrayViewMut::new(output_layout, &mut expected_output),
    )
    .expect("Leto regular forward");

    let input = device.upload(input_host).expect("input upload");
    let weight = device.upload(weight_host).expect("weight upload");
    let bias = device.upload(&bias_host).expect("bias upload");
    let output = device
        .alloc_zeroed::<f32>(grad_output_host.len())
        .expect("output allocation");
    let operations = WgpuConvolutionOps;
    let prepared = operations
        .prepare_convolution_forward(
            device,
            ConvolutionForwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                bias: Some(StridedView::new(&bias, &bias_layout)),
                output: StridedView::new(&output, &output_layout),
            },
            parameters,
        )
        .expect("prepare regular forward");
    <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::dispatch_convolution_forward::<R, S>(
        &operations,
        device,
        &prepared,
    )
    .expect("dispatch regular forward");
    assert_download_eq(device, &output, &expected_output);

    let initial_input_gradient = vec![0.25_f32; input_host.len()];
    let initial_weight_gradient = vec![-0.5_f32; weight_host.len()];
    let initial_bias_gradient = [1.5_f32];
    let mut expected_input_gradient = initial_input_gradient.clone();
    let mut expected_weight_gradient = initial_weight_gradient.clone();
    let mut expected_bias_gradient = initial_bias_gradient;
    convolution_backward_accumulate(
        &ArrayView::new(input_layout, input_host),
        &ArrayView::new(weight_layout, weight_host),
        &ArrayView::new(output_layout, grad_output_host),
        parameters,
        Some(&mut ArrayViewMut::new(
            input_layout,
            &mut expected_input_gradient,
        )),
        Some(&mut ArrayViewMut::new(
            weight_layout,
            &mut expected_weight_gradient,
        )),
        Some(&mut ArrayViewMut::new(
            bias_layout,
            &mut expected_bias_gradient,
        )),
    )
    .expect("Leto regular backward");

    let grad_output = device
        .upload(grad_output_host)
        .expect("gradient output upload");
    let grad_input = device
        .upload(&initial_input_gradient)
        .expect("input gradient upload");
    let grad_weight = device
        .upload(&initial_weight_gradient)
        .expect("weight gradient upload");
    let grad_bias = device
        .upload(&initial_bias_gradient)
        .expect("bias gradient upload");
    let prepared = operations
        .prepare_convolution_backward(
            device,
            ConvolutionBackwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                grad_output: StridedView::new(&grad_output, &output_layout),
                gradients: ConvolutionGradientViews {
                    input: Some(StridedView::new(&grad_input, &input_layout)),
                    weight: Some(StridedView::new(&grad_weight, &weight_layout)),
                    bias: Some(StridedView::new(&grad_bias, &bias_layout)),
                },
            },
            parameters,
        )
        .expect("prepare regular backward");
    <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::dispatch_convolution_backward::<R, S>(
        &operations,
        device,
        &prepared,
    )
    .expect("dispatch regular backward");
    assert_download_eq(device, &grad_input, &expected_input_gradient);
    assert_download_eq(device, &grad_weight, &expected_weight_gradient);
    assert_download_eq(device, &grad_bias, &expected_bias_gradient);
}

#[expect(
    clippy::too_many_arguments,
    reason = "the test driver receives one complete convolution case"
)]
fn verify_transposed_case<const R: usize, const S: usize>(
    device: &WgpuDevice,
    input_host: &[f32],
    input_layout: Layout<R>,
    weight_host: &[f32],
    weight_layout: Layout<R>,
    grad_output_host: &[f32],
    output_layout: Layout<R>,
    parameters: TransposedConvolutionParameters<S>,
) {
    let bias_host = [-1.0_f32];
    let bias_layout = Layout::try_new([1], [1], 0)
        .expect("invariant: submatrix layout derives from a validated parent");
    let mut expected_output = vec![0.0_f32; grad_output_host.len()];
    convolution_transposed_forward_into(
        &ArrayView::new(input_layout, input_host),
        &ArrayView::new(weight_layout, weight_host),
        Some(&ArrayView::new(bias_layout, &bias_host)),
        parameters,
        &mut ArrayViewMut::new(output_layout, &mut expected_output),
    )
    .expect("Leto transposed forward");

    let input = device.upload(input_host).expect("input upload");
    let weight = device.upload(weight_host).expect("weight upload");
    let bias = device.upload(&bias_host).expect("bias upload");
    let output = device
        .alloc_zeroed::<f32>(grad_output_host.len())
        .expect("output allocation");
    let operations = WgpuConvolutionOps;
    let prepared = operations
        .prepare_convolution_transposed_forward(
            device,
            ConvolutionForwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                bias: Some(StridedView::new(&bias, &bias_layout)),
                output: StridedView::new(&output, &output_layout),
            },
            parameters,
        )
        .expect("prepare transposed forward");
    <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::dispatch_convolution_transposed_forward::<
        R,
        S,
    >(&operations, device, &prepared)
    .expect("dispatch transposed forward");
    assert_download_eq(device, &output, &expected_output);

    let initial_input_gradient = vec![0.25_f32; input_host.len()];
    let initial_weight_gradient = vec![-0.5_f32; weight_host.len()];
    let initial_bias_gradient = [1.5_f32];
    let mut expected_input_gradient = initial_input_gradient.clone();
    let mut expected_weight_gradient = initial_weight_gradient.clone();
    let mut expected_bias_gradient = initial_bias_gradient;
    let mut expected_input_view = ArrayViewMut::new(input_layout, &mut expected_input_gradient);
    let mut expected_weight_view = ArrayViewMut::new(weight_layout, &mut expected_weight_gradient);
    let mut expected_bias_view = ArrayViewMut::new(bias_layout, &mut expected_bias_gradient);
    convolution_transposed_backward_accumulate(
        &ArrayView::new(input_layout, input_host),
        &ArrayView::new(weight_layout, weight_host),
        &ArrayView::new(output_layout, grad_output_host),
        parameters,
        TransposedConvolutionGradients::new(
            Some(&mut expected_input_view),
            Some(&mut expected_weight_view),
            Some(&mut expected_bias_view),
        ),
    )
    .expect("Leto transposed backward");

    let grad_output = device
        .upload(grad_output_host)
        .expect("gradient output upload");
    let grad_input = device
        .upload(&initial_input_gradient)
        .expect("input gradient upload");
    let grad_weight = device
        .upload(&initial_weight_gradient)
        .expect("weight gradient upload");
    let grad_bias = device
        .upload(&initial_bias_gradient)
        .expect("bias gradient upload");
    let prepared = operations
        .prepare_convolution_transposed_backward(
            device,
            ConvolutionBackwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                grad_output: StridedView::new(&grad_output, &output_layout),
                gradients: ConvolutionGradientViews {
                    input: Some(StridedView::new(&grad_input, &input_layout)),
                    weight: Some(StridedView::new(&grad_weight, &weight_layout)),
                    bias: Some(StridedView::new(&grad_bias, &bias_layout)),
                },
            },
            parameters,
        )
        .expect("prepare transposed backward");
    <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::dispatch_convolution_transposed_backward::<
        R,
        S,
    >(&operations, device, &prepared)
    .expect("dispatch transposed backward");
    assert_download_eq(device, &grad_input, &expected_input_gradient);
    assert_download_eq(device, &grad_weight, &expected_weight_gradient);
    assert_download_eq(device, &grad_bias, &expected_bias_gradient);
}

fn assert_download_eq(device: &WgpuDevice, buffer: &crate::WgpuBuffer<f32>, expected: &[f32]) {
    let mut actual = vec![0.0_f32; expected.len()];
    device
        .download(buffer, &mut actual)
        .expect("device buffer download");
    assert_eq!(actual, expected);
}
