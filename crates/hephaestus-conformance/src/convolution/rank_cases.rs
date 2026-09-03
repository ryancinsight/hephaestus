use core::fmt::Debug;

use eunomia::Pod;
use hephaestus_core::{
    ComputeDevice, ConvolutionBackwardOperands, ConvolutionForwardOperands,
    ConvolutionGradientViews, ConvolutionOps, StridedView,
};
use leto::{
    ArrayView, ArrayViewMut, ConvolutionParameters, Layout, TransposedConvolutionParameters,
};
use leto_ops::{
    Scalar, TransposedConvolutionGradients, convolution_backward_accumulate,
    convolution_forward_into, convolution_transposed_backward_accumulate,
    convolution_transposed_forward_into,
};

trait ContractScalar: Scalar + Pod + PartialEq + Debug {
    fn from_fixture(value: i8) -> Self;
}

impl ContractScalar for f32 {
    fn from_fixture(value: i8) -> Self {
        f32::from(value)
    }
}

impl ContractScalar for f64 {
    fn from_fixture(value: i8) -> Self {
        f64::from(value)
    }
}

pub(super) fn assert_higher_rank_contract<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, f32>,
{
    rank_two(device, operations);
    rank_three(device, operations);
}

pub(super) fn assert_all_rank_contract<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, f64>,
{
    rank_one(device, operations);
    rank_two(device, operations);
    rank_three(device, operations);
}

fn rank_one<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    verify_regular(
        device,
        operations,
        &fixture::<T>(&[1, -2, 3, 1]),
        Layout::try_new([1, 1, 4], [4, 4, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[2, -1]),
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -2, 3]),
        Layout::try_new([1, 1, 3], [3, 3, 1], 0).expect("valid conformance fixture layout"),
        ConvolutionParameters::new([1], [0], [1]).expect("valid rank-one parameters"),
    );
    verify_transposed(
        device,
        operations,
        &fixture::<T>(&[1, -2, 3]),
        Layout::try_new([1, 1, 3], [3, 3, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, 2]),
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, 2, -1, 1, 3]),
        Layout::try_new([1, 1, 5], [5, 5, 1], 0).expect("valid conformance fixture layout"),
        TransposedConvolutionParameters::new([2], [1], [1], [1])
            .expect("valid rank-one transposed parameters"),
    );
}

fn rank_two<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    verify_regular(
        device,
        operations,
        &fixture::<T>(&[1, -2, 3, 1, 4, -1, 2, 1, -3]),
        Layout::try_new([1, 1, 3, 3], [9, 9, 3, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -1, 2, 1]),
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -2, 1, 3]),
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0).expect("valid conformance fixture layout"),
        ConvolutionParameters::new([1, 1], [0, 0], [1, 1]).expect("valid rank-two parameters"),
    );
    verify_transposed(
        device,
        operations,
        &fixture::<T>(&[1, -2, 1, 3]),
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -1, 2, 1]),
        Layout::try_new([1, 1, 2, 2], [4, 4, 2, 1], 0).expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -2, 1, 3, -1, 2, 1, -1, 4]),
        Layout::try_new([1, 1, 3, 3], [9, 9, 3, 1], 0).expect("valid conformance fixture layout"),
        TransposedConvolutionParameters::new([2, 1], [1, 0], [1, 0], [1, 1])
            .expect("valid rank-two transposed parameters"),
    );
}

fn rank_three<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    verify_regular(
        device,
        operations,
        &fixture::<T>(&[
            1, -2, 3, 1, 4, -1, 2, 1, -3, 1, -1, 2, 1, -1, 1, 3, -2, 1, -1, 2, -3, 4, -1, 1, 1, -2,
            3,
        ]),
        Layout::try_new([1, 1, 3, 3, 3], [27, 27, 9, 3, 1], 0)
            .expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -1, 2, 1, -1, 1, -2, 1]),
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -2, 1, 3, -1, 2, 1, -1]),
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("valid conformance fixture layout"),
        ConvolutionParameters::new([1, 1, 1], [0, 0, 0], [1, 1, 1])
            .expect("valid rank-three parameters"),
    );
    verify_transposed(
        device,
        operations,
        &fixture::<T>(&[1, -2, 1, 3, -1, 2, 1, -1]),
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("valid conformance fixture layout"),
        &fixture::<T>(&[1, -1, 2, 1, -1, 1, -2, 1]),
        Layout::try_new([1, 1, 2, 2, 2], [8, 8, 4, 2, 1], 0)
            .expect("valid conformance fixture layout"),
        &fixture::<T>(&[
            1, -2, 1, 3, -1, 2, 1, -1, 4, -1, 1, -2, 3, 1, -1, 2, -3, 4, 1, -1, 1, -1, 1, -1, 1,
            -1, 1,
        ]),
        Layout::try_new([1, 1, 3, 3, 3], [27, 27, 9, 3, 1], 0)
            .expect("valid conformance fixture layout"),
        TransposedConvolutionParameters::new([2, 1, 2], [1, 0, 1], [1, 0, 1], [1, 1, 1])
            .expect("valid rank-three transposed parameters"),
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the conformance driver receives one complete convolution case"
)]
fn verify_regular<D, O, T, const R: usize, const S: usize>(
    device: &D,
    operations: &O,
    input_host: &[T],
    input_layout: Layout<R>,
    weight_host: &[T],
    weight_layout: Layout<R>,
    grad_output_host: &[T],
    output_layout: Layout<R>,
    parameters: ConvolutionParameters<S>,
) where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    let mut expected_output = vec![T::from_fixture(0); grad_output_host.len()];
    convolution_forward_into(
        &ArrayView::new(input_layout, input_host),
        &ArrayView::new(weight_layout, weight_host),
        None,
        parameters,
        &mut ArrayViewMut::new(output_layout, &mut expected_output),
    )
    .expect("Leto regular forward oracle");

    let input = device.upload(input_host).expect("input upload");
    let weight = device.upload(weight_host).expect("weight upload");
    let output = device
        .alloc_zeroed::<T>(grad_output_host.len())
        .expect("output allocation");
    operations
        .convolution_forward_into(
            device,
            ConvolutionForwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                bias: None,
                output: StridedView::new(&output, &output_layout),
            },
            parameters,
        )
        .expect("regular forward dispatch");
    assert_download_eq(device, &output, &expected_output, "regular forward");

    let initial_input_gradient = vec![T::from_fixture(1); input_host.len()];
    let initial_weight_gradient = vec![T::from_fixture(-1); weight_host.len()];
    let mut expected_input_gradient = initial_input_gradient.clone();
    let mut expected_weight_gradient = initial_weight_gradient.clone();
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
        None,
    )
    .expect("Leto regular backward oracle");

    let grad_output = device.upload(grad_output_host).expect("gradient upload");
    let grad_input = device
        .upload(&initial_input_gradient)
        .expect("input-gradient upload");
    let grad_weight = device
        .upload(&initial_weight_gradient)
        .expect("weight-gradient upload");
    operations
        .convolution_backward_accumulate(
            device,
            ConvolutionBackwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                grad_output: StridedView::new(&grad_output, &output_layout),
                gradients: ConvolutionGradientViews {
                    input: Some(StridedView::new(&grad_input, &input_layout)),
                    weight: Some(StridedView::new(&grad_weight, &weight_layout)),
                    bias: None,
                },
            },
            parameters,
        )
        .expect("regular backward dispatch");
    assert_download_eq(
        device,
        &grad_input,
        &expected_input_gradient,
        "regular input gradient",
    );
    assert_download_eq(
        device,
        &grad_weight,
        &expected_weight_gradient,
        "regular weight gradient",
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the conformance driver receives one complete convolution case"
)]
fn verify_transposed<D, O, T, const R: usize, const S: usize>(
    device: &D,
    operations: &O,
    input_host: &[T],
    input_layout: Layout<R>,
    weight_host: &[T],
    weight_layout: Layout<R>,
    grad_output_host: &[T],
    output_layout: Layout<R>,
    parameters: TransposedConvolutionParameters<S>,
) where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    let mut expected_output = vec![T::from_fixture(0); grad_output_host.len()];
    convolution_transposed_forward_into(
        &ArrayView::new(input_layout, input_host),
        &ArrayView::new(weight_layout, weight_host),
        None,
        parameters,
        &mut ArrayViewMut::new(output_layout, &mut expected_output),
    )
    .expect("Leto transposed forward oracle");

    let input = device.upload(input_host).expect("input upload");
    let weight = device.upload(weight_host).expect("weight upload");
    let output = device
        .alloc_zeroed::<T>(grad_output_host.len())
        .expect("output allocation");
    operations
        .convolution_transposed_forward_into(
            device,
            ConvolutionForwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                bias: None,
                output: StridedView::new(&output, &output_layout),
            },
            parameters,
        )
        .expect("transposed forward dispatch");
    assert_download_eq(device, &output, &expected_output, "transposed forward");

    let initial_input_gradient = vec![T::from_fixture(1); input_host.len()];
    let initial_weight_gradient = vec![T::from_fixture(-1); weight_host.len()];
    let mut expected_input_gradient = initial_input_gradient.clone();
    let mut expected_weight_gradient = initial_weight_gradient.clone();
    let mut expected_input_view = ArrayViewMut::new(input_layout, &mut expected_input_gradient);
    let mut expected_weight_view = ArrayViewMut::new(weight_layout, &mut expected_weight_gradient);
    convolution_transposed_backward_accumulate(
        &ArrayView::new(input_layout, input_host),
        &ArrayView::new(weight_layout, weight_host),
        &ArrayView::new(output_layout, grad_output_host),
        parameters,
        TransposedConvolutionGradients::new(
            Some(&mut expected_input_view),
            Some(&mut expected_weight_view),
            None,
        ),
    )
    .expect("Leto transposed backward oracle");

    let grad_output = device.upload(grad_output_host).expect("gradient upload");
    let grad_input = device
        .upload(&initial_input_gradient)
        .expect("input-gradient upload");
    let grad_weight = device
        .upload(&initial_weight_gradient)
        .expect("weight-gradient upload");
    operations
        .convolution_transposed_backward_accumulate(
            device,
            ConvolutionBackwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                grad_output: StridedView::new(&grad_output, &output_layout),
                gradients: ConvolutionGradientViews {
                    input: Some(StridedView::new(&grad_input, &input_layout)),
                    weight: Some(StridedView::new(&grad_weight, &weight_layout)),
                    bias: None,
                },
            },
            parameters,
        )
        .expect("transposed backward dispatch");
    assert_download_eq(
        device,
        &grad_input,
        &expected_input_gradient,
        "transposed input gradient",
    );
    assert_download_eq(
        device,
        &grad_weight,
        &expected_weight_gradient,
        "transposed weight gradient",
    );
}

fn fixture<T: ContractScalar>(values: &[i8]) -> Vec<T> {
    values.iter().copied().map(T::from_fixture).collect()
}

fn assert_download_eq<D, T>(device: &D, buffer: &D::Buffer<T>, expected: &[T], clause: &str)
where
    D: ComputeDevice,
    T: ContractScalar,
{
    let mut actual = vec![T::from_fixture(0); expected.len()];
    device
        .download(buffer, &mut actual)
        .expect("device buffer download");
    assert_eq!(actual, expected, "{}: {clause}", device.backend_name());
}
