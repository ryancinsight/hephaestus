use core::num::NonZeroUsize;

use leto::Layout;
use themis::MemoryTier;

use super::*;
use crate::domain::attention::GroupedKeepMask;
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

fn layout(shape: [usize; 3]) -> Layout<3> {
    Layout::c_contiguous(shape).expect("valid test layout")
}

#[test]
fn plans_grouped_mask_forward_without_materialization() {
    let query_buffer = Buffer { len: 24 };
    let key_buffer = Buffer { len: 32 };
    let value_buffer = Buffer { len: 40 };
    let output_buffer = Buffer { len: 30 };
    let weights_buffer = Buffer { len: 24 };
    let mask_buffer = Buffer { len: 8 };
    let query = layout([2, 3, 4]);
    let key = layout([2, 4, 4]);
    let value = layout([2, 4, 5]);
    let output = layout([2, 3, 5]);
    let weights = layout([2, 3, 4]);
    let mask = Layout::c_contiguous([1, 4]).expect("mask layout");
    let grouped = GroupedKeepMask::new(
        StridedView::new(&mask_buffer, &mask),
        NonZeroUsize::new(2).expect("nonzero group"),
    );
    let operands = AttentionForwardOperands {
        query: StridedView::new(&query_buffer, &query),
        key: StridedView::new(&key_buffer, &key),
        value: StridedView::new(&value_buffer, &value),
        mask: super::super::AttentionMask::causal_keep(grouped),
        scale: 0.5_f32,
        output: StridedView::new(&output_buffer, &output),
        weights: StridedView::new(&weights_buffer, &weights),
    };

    let plan = plan_attention_forward(&operands, false).expect("valid attention plan");

    assert_eq!(plan.batch, 2);
    assert_eq!(plan.score_elements, 24);
    assert_eq!(plan.max_physical_offset, 39);
}

#[test]
fn rejects_colliding_output_and_preserves_exact_injective_layouts() {
    let input_buffer = Buffer { len: 6 };
    let output_buffer = Buffer { len: 9 };
    let weights_buffer = Buffer { len: 3 };
    let input = layout([1, 3, 2]);
    let key = layout([1, 1, 2]);
    let value = layout([1, 1, 2]);
    let weights = layout([1, 3, 1]);
    let colliding = Layout::new([1, 3, 2], [6, 2, 4], 0);
    let injective = Layout::new([1, 3, 2], [6, 2, 3], 0);

    let make_operands = |output_layout| AttentionForwardOperands {
        query: StridedView::new(&input_buffer, &input),
        key: StridedView::new(&input_buffer, &key),
        value: StridedView::new(&input_buffer, &value),
        mask: super::super::AttentionMask::unrestricted(),
        scale: 1.0_f32,
        output: StridedView::new(&output_buffer, output_layout),
        weights: StridedView::new(&weights_buffer, &weights),
    };

    let error = plan_attention_forward(&make_operands(&colliding), false)
        .expect_err("colliding output must fail");
    assert!(error.to_string().contains("injectively"));
    plan_attention_forward(&make_operands(&injective), false)
        .expect("exact injective output must pass");
}

#[test]
fn backward_rejects_empty_targets_and_nonfinite_scale() {
    let buffer = Buffer { len: 8 };
    let query = layout([1, 2, 2]);
    let key = layout([1, 2, 2]);
    let value = layout([1, 2, 2]);
    let weights = layout([1, 2, 2]);
    let operands = AttentionBackwardOperands {
        grad_output: StridedView::new(&buffer, &value),
        query: StridedView::new(&buffer, &query),
        key: StridedView::new(&buffer, &key),
        value: StridedView::new(&buffer, &value),
        weights: StridedView::new(&buffer, &weights),
        scale: f64::NAN,
        gradients: AttentionGradientViews {
            query: None,
            key: None,
            value: None,
        },
    };

    let error = plan_attention_backward(&operands, false).expect_err("empty targets must fail");
    assert!(error.to_string().contains("at least one gradient"));
}
