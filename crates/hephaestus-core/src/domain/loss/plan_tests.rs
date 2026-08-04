use leto::Layout;
use themis::MemoryTier;

use super::*;
use crate::domain::view::StridedView;

struct Buffer {
    len: usize,
}

impl<T> DeviceBuffer<T> for Buffer {
    fn len(&self) -> usize {
        self.len
    }

    fn tier(&self) -> MemoryTier {
        MemoryTier::Dram
    }
}

#[test]
fn plans_strided_forward_without_materialization() {
    let values = Buffer { len: 12 };
    let targets = Buffer { len: 2 };
    let loss = Buffer { len: 2 };
    let logits = Layout::new([2, 3], [6, 2], 1);
    let target_layout = Layout::c_contiguous([2]).expect("target layout");
    let loss_layout = Layout::new([1], [1], 1);
    let operands = CrossEntropyForwardOperands {
        logits: StridedView::new(&values, &logits),
        targets: StridedView::new(&targets, &target_layout),
        loss: StridedView::new(&loss, &loss_layout),
        probabilities: StridedView::new(&values, &logits),
    };

    let plan = plan_cross_entropy_forward::<f32, _, _>(&operands, false)
        .expect("valid strided forward plan");

    assert_eq!(plan.batch, 2);
    assert_eq!(plan.classes, 3);
    assert_eq!(plan.elements, 6);
    assert!(plan.probability_tolerance > 0.0);
    assert_eq!(plan.max_physical_offset, 11);
}

#[test]
fn rejects_empty_class_support_and_target_shape_mismatch() {
    let values = Buffer { len: 1 };
    let targets = Buffer { len: 1 };
    let logits = Layout::new([1, 0], [0, 1], 0);
    let target_layout = Layout::c_contiguous([1]).expect("target layout");
    let scalar = Layout::c_contiguous([1]).expect("scalar layout");
    let operands = CrossEntropyForwardOperands {
        logits: StridedView::new(&values, &logits),
        targets: StridedView::new(&targets, &target_layout),
        loss: StridedView::new(&values, &scalar),
        probabilities: StridedView::new(&values, &logits),
    };
    let error = plan_cross_entropy_forward::<f32, _, _>(&operands, false)
        .expect_err("softmax requires class support");
    assert!(error.to_string().contains("at least one class"));

    let logits = Layout::c_contiguous([2, 2]).expect("logits layout");
    let wrong_targets = Layout::c_contiguous([1]).expect("wrong targets");
    let probabilities = Layout::c_contiguous([2, 2]).expect("probability layout");
    let operands = CrossEntropyForwardOperands {
        logits: StridedView::new(&Buffer { len: 4 }, &logits),
        targets: StridedView::new(&targets, &wrong_targets),
        loss: StridedView::new(&values, &scalar),
        probabilities: StridedView::new(&Buffer { len: 4 }, &probabilities),
    };
    let error = plan_cross_entropy_forward::<f32, _, _>(&operands, false)
        .expect_err("target count must equal batch");
    assert!(
        error
            .to_string()
            .contains("targets shape [1] must equal [2]")
    );
}

#[test]
fn rejects_aliasing_and_noninjective_destinations() {
    let values = Buffer { len: 4 };
    let targets = Buffer { len: 1 };
    let logits = Layout::c_contiguous([1, 2]).expect("logits layout");
    let target_layout = Layout::c_contiguous([1]).expect("target layout");
    let scalar = Layout::c_contiguous([1]).expect("scalar layout");
    let colliding = Layout::new([1, 2], [0, 0], 0);
    let operands = CrossEntropyForwardOperands {
        logits: StridedView::new(&values, &logits),
        targets: StridedView::new(&targets, &target_layout),
        loss: StridedView::new(&values, &scalar),
        probabilities: StridedView::new(&values, &colliding),
    };
    let error = plan_cross_entropy_forward::<f32, _, _>(&operands, false)
        .expect_err("writable layout must be injective");
    assert!(error.to_string().contains("injectively"));

    let probabilities = Layout::c_contiguous([1, 2]).expect("probability layout");
    let operands = CrossEntropyForwardOperands {
        probabilities: StridedView::new(&values, &probabilities),
        ..operands
    };
    let error = plan_cross_entropy_forward::<f32, _, _>(&operands, true)
        .expect_err("backend-reported alias must fail");
    assert!(error.to_string().contains("must not alias"));
}

#[test]
fn plans_additive_backward_and_checks_address_limit() {
    let values = Buffer { len: 8 };
    let targets = Buffer { len: 2 };
    let scalar = Layout::c_contiguous([1]).expect("scalar layout");
    let matrix = Layout::c_contiguous([2, 4]).expect("matrix layout");
    let target_layout = Layout::c_contiguous([2]).expect("target layout");
    let operands = CrossEntropyBackwardOperands {
        output_gradient: StridedView::new(&values, &scalar),
        probabilities: StridedView::new(&values, &matrix),
        targets: StridedView::new(&targets, &target_layout),
        logit_gradient: StridedView::new(&values, &matrix),
    };

    let plan =
        plan_cross_entropy_backward::<f32, _, _>(&operands, false).expect("valid backward plan");
    assert_eq!(plan.elements, 8);
    assert!(plan.validate_address_limit(7).is_err());
    plan.validate_address_limit(8)
        .expect("inclusive address limit admits every plan field");
}
