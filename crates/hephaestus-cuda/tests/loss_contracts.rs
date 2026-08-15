//! CUDA instantiation of provider-owned mean cross-entropy contracts.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_cross_entropy_contract;
use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    StridedView,
};
use hephaestus_cuda::{CudaCrossEntropyOps, CudaDevice};
use leto::Layout;

fn device(clause: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA cross-entropy {clause}: device unavailable ({error})");
            None
        }
        Err(error) => panic!("CUDA cross-entropy {clause} requires a physical device: {error}"),
    }
}

#[test]
fn cuda_satisfies_shared_cross_entropy_contract() {
    let Some(device) = device("shared contract") else {
        return;
    };
    assert_cross_entropy_contract(&device, &CudaCrossEntropyOps);
}

#[test]
fn stable_forward_and_additive_backward_preserve_strided_padding() {
    let Some(device) = device("strided value contract") else {
        return;
    };
    let matrix = Layout::try_new([2, 2], [3, 1], 1).expect("valid test layout");
    let targets_layout = Layout::try_new([2], [2], 1).expect("valid test layout");
    let scalar = Layout::try_new([1], [1], 1).expect("valid test layout");
    let logits = device
        .upload(&[91.0_f32, 0.0, 0.0, 92.0, 0.0, 0.0])
        .expect("logits upload");
    let targets = device.upload(&[81_u32, 0, 82, 1]).expect("targets upload");
    let loss = device.upload(&[71.0_f32, -1.0]).expect("loss upload");
    let probabilities = device
        .upload(&[61.0_f32, -1.0, -1.0, 62.0, -1.0, -1.0])
        .expect("probabilities upload");

    CudaCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &matrix),
                targets: StridedView::new(&targets, &targets_layout),
                loss: StridedView::new(&loss, &scalar),
                probabilities: StridedView::new(&probabilities, &matrix),
            },
        )
        .expect("cross-entropy forward");

    let mut actual_loss = [0.0_f32; 2];
    let mut actual_probabilities = [0.0_f32; 6];
    device
        .download(&loss, &mut actual_loss)
        .expect("loss download");
    device
        .download(&probabilities, &mut actual_probabilities)
        .expect("probabilities download");
    assert_eq!(actual_loss[0], 71.0);
    assert!((actual_loss[1] - core::f32::consts::LN_2).abs() <= 2.0 * f32::EPSILON);
    assert_eq!(actual_probabilities, [61.0, 0.5, 0.5, 62.0, 0.5, 0.5]);

    let output_gradient = device
        .upload(&[51.0_f32, 1.0])
        .expect("output gradient upload");
    let logit_gradient = device
        .upload(&[41.0_f32, 5.0, 5.0, 42.0, 5.0, 5.0])
        .expect("logit gradient upload");
    CudaCrossEntropyOps
        .cross_entropy_backward_accumulate(
            &device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&output_gradient, &scalar),
                probabilities: StridedView::new(&probabilities, &matrix),
                targets: StridedView::new(&targets, &targets_layout),
                logit_gradient: StridedView::new(&logit_gradient, &matrix),
            },
        )
        .expect("cross-entropy backward");

    let mut actual_gradient = [0.0_f32; 6];
    device
        .download(&logit_gradient, &mut actual_gradient)
        .expect("logit gradient download");
    assert_eq!(actual_gradient, [41.0, 4.75, 5.25, 42.0, 5.25, 4.75]);
}

#[test]
fn invalid_device_target_fails_before_output_mutation() {
    let Some(device) = device("target preflight") else {
        return;
    };
    let matrix = Layout::c_contiguous([1, 2]).expect("matrix layout");
    let vector = Layout::c_contiguous([1]).expect("vector layout");
    let logits = device.upload(&[0.0_f32, 0.0]).expect("logits upload");
    let targets = device.upload(&[2_u32]).expect("targets upload");
    let loss = device.upload(&[7.0_f32]).expect("loss upload");
    let probabilities = device
        .upload(&[8.0_f32, 9.0])
        .expect("probabilities upload");

    let error = CudaCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &matrix),
                targets: StridedView::new(&targets, &vector),
                loss: StridedView::new(&loss, &vector),
                probabilities: StridedView::new(&probabilities, &matrix),
            },
        )
        .expect_err("out-of-range target must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: cross-entropy target is outside the class dimension"
    );
    let mut actual_loss = [0.0_f32];
    let mut actual_probabilities = [0.0_f32; 2];
    device
        .download(&loss, &mut actual_loss)
        .expect("loss download");
    device
        .download(&probabilities, &mut actual_probabilities)
        .expect("probabilities download");
    assert_eq!(actual_loss, [7.0]);
    assert_eq!(actual_probabilities, [8.0, 9.0]);
}
