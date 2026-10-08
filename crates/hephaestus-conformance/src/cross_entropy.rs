//! Contract clauses for provider-owned mean cross-entropy.

use eunomia::{Pod, RealField};
use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    CrossEntropyScalar, StridedView,
};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{RealScalar, cross_entropy_backward_accumulate, cross_entropy_forward_into};

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
    assert_cross_entropy_contract_for::<D, O, f32>(device, operations);
}

/// Run strided forward and additive-backward clauses at f64 against one backend.
///
/// Mirrors [`assert_cross_entropy_contract`] with the f64 Leto oracle; every
/// clause body is shared through the width-generic driver below.
///
/// # Panics
///
/// Panics with the backend and violated clause when provider results diverge
/// from the Leto CPU contract.
pub fn assert_cross_entropy_contract_f64<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f64>,
{
    assert_cross_entropy_contract_for::<D, O, f64>(device, operations);
}

/// Run the additive-backward clauses against one backend, without forward.
///
/// Backends whose forward path is driver-limited (WGSL f64 needs `log`, which
/// aborts shader compilation on some Vulkan drivers) still prove their
/// backward path against the Leto oracle through this entry point. The
/// backward body is the same one the full contract runs; only the probability
/// input differs (Leto oracle output instead of the device forward output).
///
/// # Panics
///
/// Panics with the backend and violated clause when provider results diverge
/// from the Leto CPU contract.
pub fn assert_cross_entropy_backward_contract<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f32>,
{
    assert_cross_entropy_backward_contract_for::<D, O, f32>(device, operations);
}

/// Run the additive-backward clauses at f64 against one backend, without forward.
///
/// Mirrors [`assert_cross_entropy_backward_contract`] with the f64 Leto oracle.
///
/// # Panics
///
/// Panics with the backend and violated clause when provider results diverge
/// from the Leto CPU contract.
pub fn assert_cross_entropy_backward_contract_f64<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, f64>,
{
    assert_cross_entropy_backward_contract_for::<D, O, f64>(device, operations);
}

fn assert_cross_entropy_contract_for<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, T>,
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let oracle = strided_oracle::<T>();
    // The full contract chains device forward output into backward, so a
    // forward-output-format mismatch fails here rather than in isolation.
    let (probabilities, targets) = strided_forward::<D, O, T>(device, operations, &oracle);
    strided_backward::<D, O, T>(device, operations, &oracle, &probabilities, &targets);
    target_element_uses_only_its_executed_candidate::<D, O, T>(device, operations);
    target_failure_precedes_nonfinite_upstream::<D, O, T>(device, operations);
    invalid_probability_precedes_later_arithmetic::<D, O, T>(device, operations);
}

fn assert_cross_entropy_backward_contract_for<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, T>,
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let oracle = strided_oracle::<T>();
    let probabilities = device
        .upload(&oracle.expected_probabilities)
        .expect("probability upload");
    let targets = device.upload(&oracle.targets_host).expect("targets upload");
    strided_backward::<D, O, T>(device, operations, &oracle, &probabilities, &targets);
    target_element_uses_only_its_executed_candidate::<D, O, T>(device, operations);
    target_failure_precedes_nonfinite_upstream::<D, O, T>(device, operations);
    invalid_probability_precedes_later_arithmetic::<D, O, T>(device, operations);
}

/// Strided fixture layouts and host vectors with both Leto oracles applied.
struct StridedOracle<T> {
    logits_layout: Layout<2>,
    targets_layout: Layout<1>,
    loss_layout: Layout<1>,
    probability_layout: Layout<2>,
    logits_host: [T; 9],
    targets_host: [u32; 4],
    output_gradient_host: [T; 3],
    initial_gradient: [T; 9],
    expected_loss: [T; 3],
    expected_probabilities: [T; 9],
    expected_gradient: [T; 9],
}

fn strided_oracle<T>() -> StridedOracle<T>
where
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let logits_host = [91.0, 0.0, 2.0, 1.0, 92.0, -1.0, 3.0, 0.0, 93.0].map(T::from_f32);
    let targets_host = [7_u32, 2, 7, 0];
    let logits_layout =
        Layout::try_new([2, 3], [4, 1], 1).expect("valid conformance fixture layout");
    let targets_layout = Layout::try_new([2], [2], 1).expect("valid conformance fixture layout");
    let loss_layout = Layout::try_new([1], [2], 1).expect("valid conformance fixture layout");
    let probability_layout =
        Layout::try_new([2, 3], [4, 1], 1).expect("valid conformance fixture layout");

    let mut expected_loss = [-7.0; 3].map(T::from_f32);
    let mut expected_probabilities = [-8.0; 9].map(T::from_f32);
    cross_entropy_forward_into(
        &ArrayView::new(logits_layout, &logits_host),
        &[2, 0],
        &mut ArrayViewMut::new(loss_layout, &mut expected_loss),
        &mut ArrayViewMut::new(probability_layout, &mut expected_probabilities),
    )
    .expect("Leto cross-entropy forward oracle");

    let output_gradient_host = [11.0, 0.75, 12.0].map(T::from_f32);
    let initial_gradient = [13.0, 0.25, -0.5, 1.0, 14.0, -1.0, 0.5, 0.75, 15.0].map(T::from_f32);
    let mut expected_gradient = initial_gradient;
    cross_entropy_backward_accumulate(
        &ArrayView::new(loss_layout, &output_gradient_host),
        &ArrayView::new(probability_layout, &expected_probabilities),
        &[2, 0],
        &mut ArrayViewMut::new(probability_layout, &mut expected_gradient),
    )
    .expect("Leto cross-entropy backward oracle");

    StridedOracle {
        logits_layout,
        targets_layout,
        loss_layout,
        probability_layout,
        logits_host,
        targets_host,
        output_gradient_host,
        initial_gradient,
        expected_loss,
        expected_probabilities,
        expected_gradient,
    }
}

fn strided_forward<D, O, T>(
    device: &D,
    operations: &O,
    oracle: &StridedOracle<T>,
) -> (D::Buffer<T>, D::Buffer<u32>)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, T>,
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let logits = device.upload(&oracle.logits_host).expect("logits upload");
    let targets = device.upload(&oracle.targets_host).expect("targets upload");
    let loss = device
        .upload(&[-7.0; 3].map(T::from_f32))
        .expect("loss upload");
    let probabilities = device
        .upload(&[-8.0; 9].map(T::from_f32))
        .expect("probability upload");
    operations
        .cross_entropy_forward_into(
            device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &oracle.logits_layout),
                targets: StridedView::new(&targets, &oracle.targets_layout),
                loss: StridedView::new(&loss, &oracle.loss_layout),
                probabilities: StridedView::new(&probabilities, &oracle.probability_layout),
            },
        )
        .expect("cross-entropy forward dispatch");
    assert_close(
        device,
        &loss,
        &oracle.expected_loss,
        "cross-entropy mean loss",
    );
    assert_close(
        device,
        &probabilities,
        &oracle.expected_probabilities,
        "cross-entropy probabilities",
    );
    (probabilities, targets)
}

fn strided_backward<D, O, T>(
    device: &D,
    operations: &O,
    oracle: &StridedOracle<T>,
    probabilities: &D::Buffer<T>,
    targets: &D::Buffer<u32>,
) where
    D: ComputeDevice,
    O: CrossEntropyOps<D, T>,
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let output_gradient = device
        .upload(&oracle.output_gradient_host)
        .expect("output-gradient upload");
    let logit_gradient = device
        .upload(&oracle.initial_gradient)
        .expect("logit-gradient upload");
    operations
        .cross_entropy_backward_accumulate(
            device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&output_gradient, &oracle.loss_layout),
                probabilities: StridedView::new(probabilities, &oracle.probability_layout),
                targets: StridedView::new(targets, &oracle.targets_layout),
                logit_gradient: StridedView::new(&logit_gradient, &oracle.probability_layout),
            },
        )
        .expect("cross-entropy backward dispatch");
    assert_close(
        device,
        &logit_gradient,
        &oracle.expected_gradient,
        "additive cross-entropy gradient",
    );
}

fn target_element_uses_only_its_executed_candidate<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, T>,
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let scalar = Layout::try_new([1], [1], 0).expect("valid conformance fixture layout");
    let matrix = Layout::try_new([1, 1], [1, 1], 0).expect("valid conformance fixture layout");
    let upstream = device.upload(&[T::FINITE_MAX]).expect("upstream upload");
    let probabilities = device.upload(&[T::ONE]).expect("probability upload");
    let targets = device.upload(&[0_u32]).expect("target upload");
    let destination = device.upload(&[T::FINITE_MAX]).expect("destination upload");
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
        &[T::FINITE_MAX],
        "one-class zero increment",
    );
}

fn target_failure_precedes_nonfinite_upstream<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, T>,
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let scalar = Layout::try_new([1], [1], 0).expect("valid conformance fixture layout");
    let matrix = Layout::try_new([1, 1], [1, 1], 0).expect("valid conformance fixture layout");
    let upstream = device.upload(&[T::NAN]).expect("upstream upload");
    let probabilities = device.upload(&[T::ONE]).expect("probability upload");
    let targets = device.upload(&[1_u32]).expect("target upload");
    let destination = device
        .upload(&[T::from_f32(17.0)])
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
        .expect_err("combined semantic failures must use canonical priority");
    assert_eq!(
        error.to_string(),
        "invalid configuration: cross-entropy target is outside the class dimension"
    );
    assert_close(
        device,
        &destination,
        &[T::from_f32(17.0)],
        "combined-failure atomicity",
    );
}

fn invalid_probability_precedes_later_arithmetic<D, O, T>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: CrossEntropyOps<D, T>,
    T: CrossEntropyScalar + RealScalar + RealField,
{
    let scalar = Layout::try_new([1], [1], 0).expect("valid conformance fixture layout");
    let matrix = Layout::try_new([1, 2], [2, 1], 0).expect("valid conformance fixture layout");
    let upstream = device.upload(&[T::FINITE_MAX]).expect("upstream upload");
    let probabilities = device
        .upload(&[T::ONE, T::NAN])
        .expect("probability upload");
    let targets = device.upload(&[1_u32]).expect("target upload");
    let destination = device
        .upload(&[T::FINITE_MAX, T::from_f32(19.0)])
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
        &[T::FINITE_MAX, T::from_f32(19.0)],
        "row-failure atomicity",
    );
}

fn assert_close<D, T>(device: &D, buffer: &D::Buffer<T>, expected: &[T], clause: &str)
where
    D: ComputeDevice,
    T: RealField + Pod,
{
    let actual = device.download_owned(buffer).expect(clause);
    assert_eq!(actual.len(), expected.len(), "{clause}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual == expected {
            continue;
        }
        // One exp/log evaluation and a class-width-three reduction are bounded
        // by a small multiple of the width's epsilon; scaling by the value
        // magnitude preserves the relative bound near the additive destination.
        let magnitude = expected.abs();
        let scale = if magnitude > T::ONE {
            magnitude
        } else {
            T::ONE
        };
        let tolerance = T::from_f32(16.0) * T::EPSILON * scale;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{}: {clause}[{index}] expected {expected:?}, got {actual:?}, tolerance {tolerance:?}",
            device.backend_name()
        );
    }
}
