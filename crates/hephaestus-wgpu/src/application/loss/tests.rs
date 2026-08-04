use hephaestus_conformance::assert_cross_entropy_contract;
use hephaestus_core::{
    ComputeDevice, CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps,
    HephaestusError, StridedView,
};
use leto::Layout;

use super::WgpuCrossEntropyOps;
use crate::WgpuDevice;

#[test]
fn wgpu_satisfies_shared_cross_entropy_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_cross_entropy_contract(&device, &WgpuCrossEntropyOps);
}

#[test]
fn strided_forward_and_additive_backward_match_analytical_values() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let matrix_layout = Layout::new([2, 3], [4, 1], 1);
    let target_layout = Layout::new([2], [2], 1);
    let scalar_layout = Layout::new([1], [2], 1);
    let logits_host = [91.0, 1.0, 2.0, 3.0, 91.0, -1.0, 0.0, 2.0, 91.0];
    let targets_host = [99_u32, 2, 99, 0];
    let loss_initial = [71.0_f32, -5.0, 71.0];
    let probabilities_initial = [37.0_f32; 9];
    let logits = device.upload(&logits_host).expect("logits upload");
    let targets = device.upload(&targets_host).expect("targets upload");
    let loss = device.upload(&loss_initial).expect("loss upload");
    let probabilities = device
        .upload(&probabilities_initial)
        .expect("probabilities upload");

    WgpuCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &matrix_layout),
                targets: StridedView::new(&targets, &target_layout),
                loss: StridedView::new(&loss, &scalar_layout),
                probabilities: StridedView::new(&probabilities, &matrix_layout),
            },
        )
        .expect("strided WGPU forward");

    let expected_rows = [[1.0_f32, 2.0, 3.0], [-1.0, 0.0, 2.0]];
    let mut expected_probabilities = [[0.0_f32; 3]; 2];
    let mut expected_loss = 0.0_f32;
    for (row, values) in expected_rows.into_iter().enumerate() {
        let maximum = values.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = values
            .into_iter()
            .map(|value| (value - maximum).exp())
            .sum();
        for (class, value) in values.into_iter().enumerate() {
            expected_probabilities[row][class] = (value - maximum).exp() / denominator;
        }
        expected_loss += denominator.ln() + (maximum - values[targets_host[1 + row * 2] as usize]);
    }
    expected_loss /= 2.0;

    let mut actual_loss = [0.0_f32; 3];
    let mut actual_probabilities = [0.0_f32; 9];
    device
        .download(&loss, &mut actual_loss)
        .expect("loss download");
    device
        .download(&probabilities, &mut actual_probabilities)
        .expect("probabilities download");
    assert_eq!(actual_loss[0], loss_initial[0]);
    assert_close(actual_loss[1], expected_loss, 24);
    assert_eq!(actual_loss[2], loss_initial[2]);
    for (row, expected_row) in expected_probabilities.iter().enumerate() {
        for (class, &expected) in expected_row.iter().enumerate() {
            let index = 1 + row * 4 + class;
            assert_close(actual_probabilities[index], expected, 24);
        }
    }
    assert_eq!(actual_probabilities[0], probabilities_initial[0]);
    assert_eq!(actual_probabilities[4], probabilities_initial[4]);
    assert_eq!(actual_probabilities[8], probabilities_initial[8]);

    let upstream_host = [61.0_f32, 1.5, 61.0];
    let gradient_initial = [13.0_f32, 0.5, 1.0, 1.5, 13.0, 2.0, 2.5, 3.0, 13.0];
    let upstream = device.upload(&upstream_host).expect("upstream upload");
    let gradient = device.upload(&gradient_initial).expect("gradient upload");
    WgpuCrossEntropyOps
        .cross_entropy_backward_accumulate(
            &device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&upstream, &scalar_layout),
                probabilities: StridedView::new(&probabilities, &matrix_layout),
                targets: StridedView::new(&targets, &target_layout),
                logit_gradient: StridedView::new(&gradient, &matrix_layout),
            },
        )
        .expect("strided WGPU backward");
    let mut actual_gradient = [0.0_f32; 9];
    device
        .download(&gradient, &mut actual_gradient)
        .expect("gradient download");
    for (row, expected_row) in expected_probabilities.iter().enumerate() {
        for (class, &probability) in expected_row.iter().enumerate() {
            let index = 1 + row * 4 + class;
            let indicator = if class == targets_host[1 + row * 2] as usize {
                1.0
            } else {
                0.0
            };
            let expected = gradient_initial[index] + 1.5 * (probability - indicator) / 2.0;
            assert_close(actual_gradient[index], expected, 32);
        }
    }
    assert_eq!(actual_gradient[0], gradient_initial[0]);
    assert_eq!(actual_gradient[4], gradient_initial[4]);
    assert_eq!(actual_gradient[8], gradient_initial[8]);
}

#[test]
fn forward_mean_does_not_overflow_representable_row_losses() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let matrix_layout = Layout::c_contiguous([2, 2]).expect("matrix layout");
    let target_layout = Layout::c_contiguous([2]).expect("target layout");
    let scalar_layout = Layout::c_contiguous([1]).expect("scalar layout");
    let logits = device
        .upload(&[-1.0e38_f32, 1.0e38, -1.0e38, 1.0e38])
        .expect("logits upload");
    let targets = device.upload(&[0_u32, 0]).expect("targets upload");
    let loss = device.alloc_zeroed::<f32>(1).expect("loss allocation");
    let probabilities = device
        .alloc_zeroed::<f32>(4)
        .expect("probability allocation");
    WgpuCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &matrix_layout),
                targets: StridedView::new(&targets, &target_layout),
                loss: StridedView::new(&loss, &scalar_layout),
                probabilities: StridedView::new(&probabilities, &matrix_layout),
            },
        )
        .expect("overflow-safe mean forward");
    let mut actual = [0.0_f32];
    device.download(&loss, &mut actual).expect("loss download");
    assert!(actual[0].is_finite());
    assert_close(actual[0], 2.0e38, 8);
}

#[test]
fn invalid_device_target_preserves_forward_outputs() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let matrix_layout = Layout::c_contiguous([1, 2]).expect("matrix layout");
    let vector_layout = Layout::c_contiguous([1]).expect("vector layout");
    let logits = device.upload(&[1.0_f32, 2.0]).expect("logits upload");
    let targets = device.upload(&[2_u32]).expect("target upload");
    let loss = device.upload(&[17.0_f32]).expect("loss upload");
    let probabilities = device
        .upload(&[19.0_f32, 23.0])
        .expect("probability upload");
    let error = WgpuCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &matrix_layout),
                targets: StridedView::new(&targets, &vector_layout),
                loss: StridedView::new(&loss, &vector_layout),
                probabilities: StridedView::new(&probabilities, &matrix_layout),
            },
        )
        .expect_err("invalid target must fail preflight");
    assert_invalid(error, "cross-entropy target is outside the class dimension");
    let mut actual_loss = [0.0_f32];
    let mut actual_probabilities = [0.0_f32; 2];
    device
        .download(&loss, &mut actual_loss)
        .expect("loss download");
    device
        .download(&probabilities, &mut actual_probabilities)
        .expect("probability download");
    assert_eq!(actual_loss, [17.0]);
    assert_eq!(actual_probabilities, [19.0, 23.0]);
}

#[test]
fn invalid_probabilities_preserve_additive_destination() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let matrix_layout = Layout::c_contiguous([1, 2]).expect("matrix layout");
    let vector_layout = Layout::c_contiguous([1]).expect("vector layout");
    let upstream = device.upload(&[1.0_f32]).expect("upstream upload");
    let probabilities = device.upload(&[0.75_f32, 0.0]).expect("probability upload");
    let targets = device.upload(&[0_u32]).expect("target upload");
    let gradient = device.upload(&[29.0_f32, 31.0]).expect("gradient upload");
    let error = WgpuCrossEntropyOps
        .cross_entropy_backward_accumulate(
            &device,
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(&upstream, &vector_layout),
                probabilities: StridedView::new(&probabilities, &matrix_layout),
                targets: StridedView::new(&targets, &vector_layout),
                logit_gradient: StridedView::new(&gradient, &matrix_layout),
            },
        )
        .expect_err("invalid probabilities must fail preflight");
    assert_invalid(
        error,
        "cross-entropy saved probabilities do not form a valid row",
    );
    let mut actual = [0.0_f32; 2];
    device
        .download(&gradient, &mut actual)
        .expect("gradient download");
    assert_eq!(actual, [29.0, 31.0]);
}

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-cross-entropy-test") {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping WGPU cross-entropy test: {error}");
            None
        }
    }
}

fn assert_close(actual: f32, expected: f32, operation_count: u16) {
    let roundoff = f32::from(operation_count) * f32::EPSILON;
    let gamma = roundoff / (1.0 - roundoff);
    let bound = 2.0 * gamma * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= bound,
        "actual {actual}, expected {expected}, derived bound {bound}"
    );
}

fn assert_invalid(error: HephaestusError, expected: &str) {
    match error {
        HephaestusError::InvalidConfiguration { message } => assert_eq!(message, expected),
        other => panic!("expected invalid configuration, got {other:?}"),
    }
}
