//! Contract clauses for provider-owned convolution.
//!
//! The fixtures use exactly representable integer-valued `f32` arithmetic, so
//! backend reduction order cannot change the expected values.

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

pub(crate) mod rank_cases;

pub use rank_cases::ContractScalar;
use rank_cases::assert_download_eq;

/// Run the shared regular, transposed, and rank-one-through-three
/// convolution clauses at scalar `T`.
///
/// Every clause body is written once against [`ContractScalar`] and
/// monomorphized per width; callers select the width with turbofish
/// (`_contract::<_, _, f64>`). Both widths run the same clause set.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// provider contract.
pub fn assert_convolution_contract<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    regular_forward_backward::<D, O, T>(device, operations);
    transposed_forward_backward::<D, O, T>(device, operations);
    rank_cases::rank_one::<D, O, T>(device, operations);
    rank_cases::rank_two::<D, O, T>(device, operations);
    rank_cases::rank_three::<D, O, T>(device, operations);
    aliases_are_rejected_before_mutation::<D, O, T>(device, operations);
}

fn regular_forward_backward<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    let input_host = [1i8, 2, 3, 4].map(T::from_fixture);
    let weight_host = [2i8, -1].map(T::from_fixture);
    let bias_host = [1i8].map(T::from_fixture);
    let grad_output_host = [1i8, -2, 3].map(T::from_fixture);
    let input_layout =
        Layout::try_new([1, 1, 4], [4, 4, 1], 0).expect("valid conformance fixture layout");
    let weight_layout =
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout");
    let output_layout =
        Layout::try_new([1, 1, 3], [3, 3, 1], 0).expect("valid conformance fixture layout");
    let bias_layout = Layout::try_new([1], [1], 0).expect("valid conformance fixture layout");
    let parameters = ConvolutionParameters::new([1], [0], [1]).expect("valid regular parameters");

    let mut expected_output = [T::from_fixture(0); 3];
    convolution_forward_into(
        &ArrayView::new(input_layout, &input_host),
        &ArrayView::new(weight_layout, &weight_host),
        Some(&ArrayView::new(bias_layout, &bias_host)),
        parameters,
        &mut ArrayViewMut::new(output_layout, &mut expected_output),
    )
    .expect("Leto regular forward oracle");

    let input = device.upload(&input_host).expect("input upload");
    let weight = device.upload(&weight_host).expect("weight upload");
    let bias = device.upload(&bias_host).expect("bias upload");
    let output = device.alloc_zeroed::<T>(3).expect("output allocation");
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
        .expect("regular forward preparation");
    operations
        .dispatch_convolution_forward::<3, 1>(device, &prepared)
        .expect("regular forward dispatch");
    assert_download_eq(device, &output, &expected_output, "regular forward");

    let initial_input_gradient = [T::from_fixture(1); 4];
    let initial_weight_gradient = [T::from_fixture(-1); 2];
    let initial_bias_gradient = [T::from_fixture(2); 1];
    let mut expected_input_gradient = initial_input_gradient;
    let mut expected_weight_gradient = initial_weight_gradient;
    let mut expected_bias_gradient = initial_bias_gradient;
    convolution_backward_accumulate(
        &ArrayView::new(input_layout, &input_host),
        &ArrayView::new(weight_layout, &weight_host),
        &ArrayView::new(output_layout, &grad_output_host),
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
    .expect("Leto regular backward oracle");

    let grad_output = device
        .upload(&grad_output_host)
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
        .expect("regular backward preparation");
    operations
        .dispatch_convolution_backward::<3, 1>(device, &prepared)
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
    assert_download_eq(
        device,
        &grad_bias,
        &expected_bias_gradient,
        "regular bias gradient",
    );
}

fn transposed_forward_backward<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    let input_host = [1i8, -2, 3].map(T::from_fixture);
    let weight_host = [1i8, 2].map(T::from_fixture);
    let bias_host = [-1i8].map(T::from_fixture);
    let grad_output_host = [1i8, 2, -1, 1, 3].map(T::from_fixture);
    let input_layout =
        Layout::try_new([1, 1, 3], [3, 3, 1], 0).expect("valid conformance fixture layout");
    let weight_layout =
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout");
    let output_layout =
        Layout::try_new([1, 1, 5], [5, 5, 1], 0).expect("valid conformance fixture layout");
    let bias_layout = Layout::try_new([1], [1], 0).expect("valid conformance fixture layout");
    let parameters = TransposedConvolutionParameters::new([2], [1], [1], [1])
        .expect("valid transposed parameters");

    let mut expected_output = [T::from_fixture(0); 5];
    convolution_transposed_forward_into(
        &ArrayView::new(input_layout, &input_host),
        &ArrayView::new(weight_layout, &weight_host),
        Some(&ArrayView::new(bias_layout, &bias_host)),
        parameters,
        &mut ArrayViewMut::new(output_layout, &mut expected_output),
    )
    .expect("Leto transposed forward oracle");

    let input = device.upload(&input_host).expect("input upload");
    let weight = device.upload(&weight_host).expect("weight upload");
    let bias = device.upload(&bias_host).expect("bias upload");
    let output = device.alloc_zeroed::<T>(5).expect("output allocation");
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
        .expect("transposed forward preparation");
    operations
        .dispatch_convolution_transposed_forward::<3, 1>(device, &prepared)
        .expect("transposed forward dispatch");
    assert_download_eq(device, &output, &expected_output, "transposed forward");

    let initial_input_gradient = [T::from_fixture(1); 3];
    let initial_weight_gradient = [T::from_fixture(-1); 2];
    let initial_bias_gradient = [T::from_fixture(2); 1];
    let mut expected_input_gradient = initial_input_gradient;
    let mut expected_weight_gradient = initial_weight_gradient;
    let mut expected_bias_gradient = initial_bias_gradient;
    let mut expected_input_view = ArrayViewMut::new(input_layout, &mut expected_input_gradient);
    let mut expected_weight_view = ArrayViewMut::new(weight_layout, &mut expected_weight_gradient);
    let mut expected_bias_view = ArrayViewMut::new(bias_layout, &mut expected_bias_gradient);
    convolution_transposed_backward_accumulate(
        &ArrayView::new(input_layout, &input_host),
        &ArrayView::new(weight_layout, &weight_host),
        &ArrayView::new(output_layout, &grad_output_host),
        parameters,
        TransposedConvolutionGradients::new(
            Some(&mut expected_input_view),
            Some(&mut expected_weight_view),
            Some(&mut expected_bias_view),
        ),
    )
    .expect("Leto transposed backward oracle");

    let grad_output = device
        .upload(&grad_output_host)
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
        .expect("transposed backward preparation");
    operations
        .dispatch_convolution_transposed_backward::<3, 1>(device, &prepared)
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
    assert_download_eq(
        device,
        &grad_bias,
        &expected_bias_gradient,
        "transposed bias gradient",
    );
}

fn aliases_are_rejected_before_mutation<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: ConvolutionOps<D, T>,
    T: ContractScalar,
{
    let initial = [1i8, 2, 3, 4].map(T::from_fixture);
    let shared = device.upload(&initial).expect("shared buffer upload");
    let weight = device
        .upload(&[2i8, -1].map(T::from_fixture))
        .expect("weight upload");
    let input_layout =
        Layout::try_new([1, 1, 4], [4, 4, 1], 0).expect("valid conformance fixture layout");
    let weight_layout =
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout");
    let output_layout =
        Layout::try_new([1, 1, 3], [3, 3, 1], 0).expect("valid conformance fixture layout");
    let parameters = ConvolutionParameters::new([1], [0], [1]).expect("valid regular parameters");
    let result = operations.prepare_convolution_forward(
        device,
        ConvolutionForwardOperands {
            input: StridedView::new(&shared, &input_layout),
            weight: StridedView::new(&weight, &weight_layout),
            bias: None,
            output: StridedView::new(&shared, &output_layout),
        },
        parameters,
    );
    let error = match result {
        Ok(_) => panic!("aliased convolution preparation must fail"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "invalid configuration: convolution writable buffers must not alias readable operands or each other",
        "{}: alias rejection",
        device.backend_name()
    );
    assert_download_eq(
        device,
        &shared,
        &initial,
        "alias rejection preserves storage",
    );
}
