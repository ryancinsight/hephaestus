//! Host instantiation of the shared convolution clauses (`f32` and `f64`,
//! since the host's scalar bound admits both), plus host-specific cases the
//! clauses do not reach.

use hephaestus_conformance::{assert_convolution_contract, assert_convolution_f64_contract};
use hephaestus_core::{
    ComputeDevice, ConvolutionBackwardOperands, ConvolutionForwardOperands,
    ConvolutionGradientViews, ConvolutionOps,
};
use hephaestus_core::{HephaestusError, StridedView};
use hephaestus_host::{HostConvolutionOps, HostDevice};
use leto::{ConvolutionParameters, Layout};

#[test]
fn host_satisfies_the_convolution_contract() {
    assert_convolution_contract(&HostDevice::new(), &HostConvolutionOps);
}

#[test]
fn host_satisfies_the_convolution_f64_contract() {
    assert_convolution_f64_contract(&HostDevice::new(), &HostConvolutionOps);
}

/// `grad_weight` naming the same allocation as `grad_input` is illegal
/// aliasing between two writable targets; the plan rejects it before any
/// dispatch runs and neither destination is mutated.
#[test]
fn weight_gradient_aliasing_input_gradient_is_rejected() {
    let device = HostDevice::new();
    let input_layout =
        Layout::try_new([1, 1, 4], [4, 4, 1], 0).expect("valid conformance fixture layout");
    let weight_layout =
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout");
    let output_layout =
        Layout::try_new([1, 1, 3], [3, 3, 1], 0).expect("valid conformance fixture layout");
    let parameters = ConvolutionParameters::new([1], [0], [1]).expect("valid regular parameters");

    let input = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("input upload");
    let weight = device.upload(&[2.0_f32, -1.0]).expect("weight upload");
    let grad_output = device
        .upload(&[1.0_f32, -2.0, 3.0])
        .expect("gradient upload");
    let shared_gradient = device
        .upload(&[0.0_f32, 0.0, 0.0, 0.0])
        .expect("shared gradient upload");

    let error = HostConvolutionOps
        .convolution_backward_accumulate(
            &device,
            ConvolutionBackwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                grad_output: StridedView::new(&grad_output, &output_layout),
                gradients: ConvolutionGradientViews {
                    input: Some(StridedView::new(&shared_gradient, &input_layout)),
                    weight: Some(StridedView::new(&shared_gradient, &weight_layout)),
                    bias: None,
                },
            },
            parameters,
        )
        .expect_err("weight gradient aliasing input gradient must be rejected");
    assert!(
        matches!(error, HephaestusError::InvalidConfiguration { .. }),
        "{error:?}"
    );
    let mut actual = [0.0_f32; 4];
    device
        .download(&shared_gradient, &mut actual)
        .expect("gradient download");
    assert_eq!(actual, [0.0; 4], "alias rejection preserves storage");
}

/// A convolution whose output is empty (a zero spatial extent from
/// full-length padding-free striding) is a legitimate degenerate case the
/// shared clause does not exercise; leto-ops handles it as a no-op forward
/// and the host must not panic walking a zero-sized plan.
#[test]
fn zero_output_extent_forward_is_a_no_op() {
    let device = HostDevice::new();
    let input_layout = Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid fixture layout");
    let weight_layout = Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid fixture layout");
    let output_layout = Layout::try_new([1, 1, 0], [0, 0, 1], 0).expect("valid fixture layout");
    let parameters = ConvolutionParameters::new([1], [0], [1]).expect("valid regular parameters");

    let input = device.upload(&[1.0_f32]).expect("input upload");
    let weight = device.upload(&[2.0_f32, -1.0]).expect("weight upload");
    let output = device.alloc_zeroed::<f32>(0).expect("output allocation");

    HostConvolutionOps
        .convolution_forward_into(
            &device,
            ConvolutionForwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                bias: None,
                output: StridedView::new(&output, &output_layout),
            },
            parameters,
        )
        .expect("a zero-extent output must dispatch as a no-op, not panic");
}
