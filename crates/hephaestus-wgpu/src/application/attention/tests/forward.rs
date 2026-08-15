use core::num::NonZeroUsize;

use hephaestus_core::{
    AttentionForwardOperands, AttentionMask, AttentionOps, ComputeDevice, GroupedKeepMask,
    StridedView,
};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{AttentionMask as LetoAttentionMask, scaled_dot_product_attention_into};

use super::{assert_close, device_or_skip};
use crate::WgpuAttentionOps;

#[test]
fn grouped_causal_strided_forward_matches_leto_and_zeroes_fully_masked_rows() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let tensor_layout = Layout::try_new([4, 2, 2], [5, 2, 1], 1).expect("valid test layout");
    let score_layout = Layout::try_new([4, 2, 2], [5, 2, 1], 1).expect("valid test layout");
    let query_host = backing(&tensor_layout, |batch, sequence, feature| {
        0.1 + batch as f32 * 0.2 + sequence as f32 * 0.3 - feature as f32 * 0.15
    });
    let key_host = backing(&tensor_layout, |batch, sequence, feature| {
        -0.2 + batch as f32 * 0.1 + sequence as f32 * 0.25 + feature as f32 * 0.2
    });
    let value_host = backing(&tensor_layout, |batch, sequence, feature| {
        0.4 - batch as f32 * 0.05 + sequence as f32 * 0.4 + feature as f32 * 0.1
    });
    let mask_layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid test layout");
    let mask_host = [9.0_f32, 1.0, 0.0, 9.0, 0.0, 0.0];
    let output_initial = vec![7.0_f32; backing_len(&tensor_layout)];
    let weights_initial = vec![7.0_f32; backing_len(&score_layout)];

    let mut expected_output = output_initial.clone();
    let mut expected_weights = weights_initial.clone();
    let materialized_mask = [1.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let materialized_layout = Layout::c_contiguous([4, 1, 2]).expect("mask layout");
    scaled_dot_product_attention_into(
        &ArrayView::new(tensor_layout, &query_host),
        &ArrayView::new(tensor_layout, &key_host),
        &ArrayView::new(tensor_layout, &value_host),
        LetoAttentionMask::CausalKeep(ArrayView::new(materialized_layout, &materialized_mask)),
        0.75,
        &mut ArrayViewMut::new(tensor_layout, &mut expected_output),
        &mut ArrayViewMut::new(score_layout, &mut expected_weights),
    )
    .expect("Leto grouped-mask oracle");

    let query = device.upload(&query_host).expect("query upload");
    let key = device.upload(&key_host).expect("key upload");
    let value = device.upload(&value_host).expect("value upload");
    let mask = device.upload(&mask_host).expect("mask upload");
    let output = device.upload(&output_initial).expect("output upload");
    let weights = device.upload(&weights_initial).expect("weights upload");
    WgpuAttentionOps
        .attention_forward_into(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &tensor_layout),
                key: StridedView::new(&key, &tensor_layout),
                value: StridedView::new(&value, &tensor_layout),
                mask: AttentionMask::causal_keep(GroupedKeepMask::new(
                    StridedView::new(&mask, &mask_layout),
                    NonZeroUsize::new(2).expect("nonzero group width"),
                )),
                scale: 0.75,
                output: StridedView::new(&output, &tensor_layout),
                weights: StridedView::new(&weights, &score_layout),
            },
        )
        .expect("WGPU grouped-mask forward");
    let mut actual_output = vec![0.0_f32; output_initial.len()];
    let mut actual_weights = vec![0.0_f32; weights_initial.len()];
    device
        .download(&output, &mut actual_output)
        .expect("output download");
    device
        .download(&weights, &mut actual_weights)
        .expect("weights download");
    assert_close(&actual_output, &expected_output, 128);
    assert_close(&actual_weights, &expected_weights, 128);

    for batch in 2..4 {
        for sequence in 0..2 {
            for feature in 0..2 {
                assert_eq!(
                    actual_output[physical(&tensor_layout, [batch, sequence, feature])],
                    0.0
                );
                assert_eq!(
                    actual_weights[physical(&score_layout, [batch, sequence, feature])],
                    0.0
                );
            }
        }
    }
}

fn backing(layout: &Layout<3>, value: impl Fn(usize, usize, usize) -> f32) -> Vec<f32> {
    let mut storage = vec![13.0_f32; backing_len(layout)];
    for batch in 0..layout.shape()[0] {
        for sequence in 0..layout.shape()[1] {
            for feature in 0..layout.shape()[2] {
                storage[physical(layout, [batch, sequence, feature])] =
                    value(batch, sequence, feature);
            }
        }
    }
    storage
}

fn backing_len(layout: &Layout<3>) -> usize {
    layout
        .checked_min_max_offsets()
        .expect("valid test layout")
        .1
        + 1
}

fn physical(layout: &Layout<3>, index: [usize; 3]) -> usize {
    let offset = layout.offset() as isize
        + index
            .into_iter()
            .zip(layout.strides())
            .map(|(coordinate, stride)| coordinate as isize * stride)
            .sum::<isize>();
    usize::try_from(offset).expect("valid test offset")
}
