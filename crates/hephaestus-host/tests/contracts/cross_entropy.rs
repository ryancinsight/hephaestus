//! Host instantiation of the shared mean cross-entropy clause, plus
//! host-specific cases it does not reach.

use hephaestus_conformance::{assert_cross_entropy_contract, assert_cross_entropy_contract_f64};
use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    HephaestusError, StridedView,
};
use hephaestus_host::{HostCrossEntropyOps, HostDevice};
use leto::Layout;

#[test]
fn host_satisfies_the_cross_entropy_contract() {
    assert_cross_entropy_contract(&HostDevice::new(), &HostCrossEntropyOps);
}

#[test]
fn host_satisfies_the_f64_cross_entropy_contract() {
    assert_cross_entropy_contract_f64(&HostDevice::new(), &HostCrossEntropyOps);
}

/// f64 forward/backward through the generic host seam: logits `[[0, 0]]`
/// give probabilities `[0.5, 0.5]` and mean loss `ln 2`; upstream `2.0`
/// accumulates gradient `[-1, 1]`.
#[test]
fn host_f64_cross_entropy_matches_hand_derived_values() {
    let device = HostDevice::new();
    let logits_layout = Layout::c_contiguous([1, 2]).expect("valid fixture layout");
    let scalar_layout = Layout::c_contiguous([1]).expect("valid fixture layout");

    let logits = device.upload(&[0.0_f64, 0.0]).expect("logits upload");
    let targets = device.upload(&[0_u32]).expect("targets upload");
    let loss = device.alloc_zeroed::<f64>(1).expect("loss allocation");
    let probabilities = device
        .alloc_zeroed::<f64>(2)
        .expect("probability allocation");
    HostCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &logits_layout),
                targets: StridedView::new(&targets, &scalar_layout),
                loss: StridedView::new(&loss, &scalar_layout),
                probabilities: StridedView::new(&probabilities, &logits_layout),
            },
        )
        .expect("valid f64 forward");
    let mut loss_actual = [0.0_f64; 1];
    device
        .download(&loss, &mut loss_actual)
        .expect("loss download");
    assert!(
        (loss_actual[0] - 2.0_f64.ln()).abs() < 1.0e-12,
        "f64 loss {}, expected ln 2",
        loss_actual[0]
    );
    let mut probabilities_actual = [0.0_f64; 2];
    device
        .download(&probabilities, &mut probabilities_actual)
        .expect("probability download");
    assert_eq!(probabilities_actual, [0.5, 0.5]);

    let upstream = device.upload(&[2.0_f64]).expect("upstream upload");
    let logit_gradient = device.alloc_zeroed::<f64>(2).expect("gradient allocation");
    HostCrossEntropyOps
        .cross_entropy_backward_accumulate(
            &device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&upstream, &scalar_layout),
                probabilities: StridedView::new(&probabilities, &logits_layout),
                targets: StridedView::new(&targets, &scalar_layout),
                logit_gradient: StridedView::new(&logit_gradient, &logits_layout),
            },
        )
        .expect("valid f64 backward");
    let mut gradient_actual = [0.0_f64; 2];
    device
        .download(&logit_gradient, &mut gradient_actual)
        .expect("gradient download");
    assert_eq!(gradient_actual, [-1.0, 1.0]);
}

/// A non-finite logit in one row and an out-of-range target in another must
/// reduce to the shader's canonical priority (`NonFiniteLogits` before
/// `TargetOutOfRange`, mirroring `hephaestus-wgpu`'s `atomicMin`-combined
/// status), and neither destination may be written.
#[test]
fn nonfinite_logit_precedes_target_out_of_range_across_rows() {
    let device = HostDevice::new();
    let logits_layout = Layout::c_contiguous([2, 2]).expect("valid fixture layout");
    let scalar_layout = Layout::c_contiguous([2]).expect("valid fixture layout");
    let loss_layout = Layout::c_contiguous([1]).expect("valid fixture layout");

    let logits_host = [f32::NAN, 0.0, 0.0, 1.0];
    let targets_host = [0_u32, 5];
    let logits = device.upload(&logits_host).expect("logits upload");
    let targets = device.upload(&targets_host).expect("targets upload");
    let loss = device.upload(&[-7.0_f32]).expect("loss upload");
    let probabilities = device.upload(&[-8.0_f32; 4]).expect("probability upload");

    let error = HostCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &logits_layout),
                targets: StridedView::new(&targets, &scalar_layout),
                loss: StridedView::new(&loss, &loss_layout),
                probabilities: StridedView::new(&probabilities, &logits_layout),
            },
        )
        .expect_err("combined semantic failures must use canonical priority");
    assert_eq!(
        error.to_string(),
        "invalid configuration: cross-entropy logits contain a non-finite value"
    );

    let mut loss_actual = [0.0_f32; 1];
    device
        .download(&loss, &mut loss_actual)
        .expect("loss download");
    assert_eq!(loss_actual, [-7.0], "forward-failure atomicity: loss");
    let mut probabilities_actual = [0.0_f32; 4];
    device
        .download(&probabilities, &mut probabilities_actual)
        .expect("probability download");
    assert_eq!(
        probabilities_actual, [-8.0; 4],
        "forward-failure atomicity: probabilities"
    );
}

/// `probabilities` naming the same allocation as `logits` is illegal
/// aliasing between a writable destination and a readable operand; the plan
/// rejects it before any preflight or arithmetic runs.
#[test]
fn probabilities_aliasing_logits_is_rejected() {
    let device = HostDevice::new();
    let logits_layout = Layout::c_contiguous([1, 2]).expect("valid fixture layout");
    let scalar_layout = Layout::c_contiguous([1]).expect("valid fixture layout");

    let logits = device.upload(&[0.0_f32, 1.0]).expect("logits upload");
    let targets = device.upload(&[0_u32]).expect("targets upload");
    let loss = device.alloc_zeroed::<f32>(1).expect("loss allocation");

    let error = HostCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &logits_layout),
                targets: StridedView::new(&targets, &scalar_layout),
                loss: StridedView::new(&loss, &scalar_layout),
                probabilities: StridedView::new(&logits, &logits_layout),
            },
        )
        .expect_err("probabilities aliasing logits must be rejected");
    assert!(
        matches!(error, HephaestusError::InvalidConfiguration { .. }),
        "{error:?}"
    );
}
