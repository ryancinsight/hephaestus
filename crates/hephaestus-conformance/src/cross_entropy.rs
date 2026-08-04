//! Contract clauses for provider-owned mean cross-entropy.

use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    StridedView,
};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{cross_entropy_backward_accumulate, cross_entropy_forward_into};

/// Run strided forward and additive-backward clauses against one backend.
///
/// # Panics
///
/// Panics with the backend and violated clause when provider results diverge
/// from the Leto CPU contract.
pub fn assert_cross_entropy_contract<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f32>,
{
    strided_forward_and_backward(device, operations);
    target_element_uses_only_its_executed_candidate(device, operations);
    target_failure_precedes_nonfinite_upstream(device, operations);
    invalid_probability_precedes_later_arithmetic(device, operations);
}

fn strided_forward_and_backward<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f32>,
{
    let logits_host = [91.0_f32, 0.0, 2.0, 1.0, 92.0, -1.0, 3.0, 0.0, 93.0];
    let targets_host = [7_u32, 2, 7, 0];
    let logits_layout = Layout::new([2, 3], [4, 1], 1);
    let targets_layout = Layout::new([2], [2], 1);
    let loss_layout = Layout::new([1], [2], 1);
    let probability_layout = Layout::new([2, 3], [4, 1], 1);

    let mut expected_loss = [-7.0_f32; 3];
    let mut expected_probabilities = [-8.0_f32; 9];
    cross_entropy_forward_into(
        &ArrayView::new(logits_layout, &logits_host),
        &[2, 0],
        &mut ArrayViewMut::new(loss_layout, &mut expected_loss),
        &mut ArrayViewMut::new(probability_layout, &mut expected_probabilities),
    )
    .expect("Leto cross-entropy forward oracle");

    let logits = device.upload(&logits_host).expect("logits upload");
    let targets = device.upload(&targets_host).expect("targets upload");
    let loss = device.upload(&[-7.0_f32; 3]).expect("loss upload");
    let probabilities = device.upload(&[-8.0_f32; 9]).expect("probability upload");
    operations
        .cross_entropy_forward_into(
            device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &logits_layout),
                targets: StridedView::new(&targets, &targets_layout),
                loss: StridedView::new(&loss, &loss_layout),
                probabilities: StridedView::new(&probabilities, &probability_layout),
            },
        )
        .expect("cross-entropy forward dispatch");
    assert_close(device, &loss, &expected_loss, "cross-entropy mean loss");
    assert_close(
        device,
        &probabilities,
        &expected_probabilities,
        "cross-entropy probabilities",
    );

    let output_gradient_host = [11.0_f32, 0.75, 12.0];
    let output_gradient_layout = loss_layout;
    let initial_gradient = [13.0_f32, 0.25, -0.5, 1.0, 14.0, -1.0, 0.5, 0.75, 15.0];
    let mut expected_gradient = initial_gradient;
    cross_entropy_backward_accumulate(
        &ArrayView::new(output_gradient_layout, &output_gradient_host),
        &ArrayView::new(probability_layout, &expected_probabilities),
        &[2, 0],
        &mut ArrayViewMut::new(probability_layout, &mut expected_gradient),
    )
    .expect("Leto cross-entropy backward oracle");

    let output_gradient = device
        .upload(&output_gradient_host)
        .expect("output-gradient upload");
    let logit_gradient = device
        .upload(&initial_gradient)
        .expect("logit-gradient upload");
    operations
        .cross_entropy_backward_accumulate(
            device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&output_gradient, &output_gradient_layout),
                probabilities: StridedView::new(&probabilities, &probability_layout),
                targets: StridedView::new(&targets, &targets_layout),
                logit_gradient: StridedView::new(&logit_gradient, &probability_layout),
            },
        )
        .expect("cross-entropy backward dispatch");
    assert_close(
        device,
        &logit_gradient,
        &expected_gradient,
        "additive cross-entropy gradient",
    );
}

fn target_element_uses_only_its_executed_candidate<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f32>,
{
    let scalar = Layout::new([1], [1], 0);
    let matrix = Layout::new([1, 1], [1, 1], 0);
    let upstream = device.upload(&[f32::MAX]).expect("upstream upload");
    let probabilities = device.upload(&[1.0_f32]).expect("probability upload");
    let targets = device.upload(&[0_u32]).expect("target upload");
    let destination = device.upload(&[f32::MAX]).expect("destination upload");
    operations
        .cross_entropy_backward_accumulate(
            device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&upstream, &scalar),
                probabilities: StridedView::new(&probabilities, &matrix),
                targets: StridedView::new(&targets, &scalar),
                logit_gradient: StridedView::new(&destination, &matrix),
            },
        )
        .expect("zero target increment must not evaluate a non-target candidate");
    assert_close(
        device,
        &destination,
        &[f32::MAX],
        "one-class zero increment",
    );
}

fn target_failure_precedes_nonfinite_upstream<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f32>,
{
    let scalar = Layout::new([1], [1], 0);
    let matrix = Layout::new([1, 1], [1, 1], 0);
    let upstream = device.upload(&[f32::NAN]).expect("upstream upload");
    let probabilities = device.upload(&[1.0_f32]).expect("probability upload");
    let targets = device.upload(&[1_u32]).expect("target upload");
    let destination = device.upload(&[17.0_f32]).expect("destination upload");
    let error = operations
        .cross_entropy_backward_accumulate(
            device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&upstream, &scalar),
                probabilities: StridedView::new(&probabilities, &matrix),
                targets: StridedView::new(&targets, &scalar),
                logit_gradient: StridedView::new(&destination, &matrix),
            },
        )
        .expect_err("combined semantic failures must use canonical priority");
    assert_eq!(
        error.to_string(),
        "invalid configuration: cross-entropy target is outside the class dimension"
    );
    assert_close(device, &destination, &[17.0], "combined-failure atomicity");
}

fn invalid_probability_precedes_later_arithmetic<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f32>,
{
    let scalar = Layout::new([1], [1], 0);
    let matrix = Layout::new([1, 2], [2, 1], 0);
    let upstream = device.upload(&[f32::MAX]).expect("upstream upload");
    let probabilities = device
        .upload(&[1.0_f32, f32::NAN])
        .expect("probability upload");
    let targets = device.upload(&[1_u32]).expect("target upload");
    let destination = device
        .upload(&[f32::MAX, 19.0])
        .expect("destination upload");
    let error = operations
        .cross_entropy_backward_accumulate(
            device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&upstream, &scalar),
                probabilities: StridedView::new(&probabilities, &matrix),
                targets: StridedView::new(&targets, &scalar),
                logit_gradient: StridedView::new(&destination, &matrix),
            },
        )
        .expect_err("provider must reduce all row failures to canonical priority");
    assert_eq!(
        error.to_string(),
        "invalid configuration: cross-entropy saved probabilities do not form a valid row"
    );
    assert_close(
        device,
        &destination,
        &[f32::MAX, 19.0],
        "row-failure atomicity",
    );
}

fn assert_close<D>(device: &D, buffer: &D::Buffer<f32>, expected: &[f32], clause: &str)
where
    D: ComputeDevice,
{
    let actual = device.download_owned(buffer).expect(clause);
    assert_eq!(actual.len(), expected.len(), "{clause}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual == expected {
            continue;
        }
        // One exp/log evaluation and a class-width-three reduction are bounded
        // by a small multiple of f32 epsilon; scaling by the value magnitude
        // preserves the relative bound near the additive destination.
        let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{}: {clause}[{index}] expected {expected}, got {actual}, tolerance {tolerance}",
            device.backend_name()
        );
    }
}
