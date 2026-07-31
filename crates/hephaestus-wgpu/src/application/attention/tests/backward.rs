use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionGradientViews, AttentionMask,
    AttentionOps, ComputeDevice, StridedView,
};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{
    AttentionGradients, AttentionMask as LetoAttentionMask,
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
};

use super::{assert_close, device_or_skip};
use crate::WgpuAttentionOps;

#[test]
fn backward_matches_leto_and_accumulates_prefilled_gradients() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let layout = Layout::c_contiguous([1, 2, 2]).expect("tensor layout");
    let query_host = [0.2_f32, -0.3, 0.7, 0.5];
    let key_host = [0.4_f32, -0.1, -0.2, 0.8];
    let value_host = [1.2_f32, -0.7, 0.3, 0.9];
    let grad_output_host = [0.5_f32, -0.4, 0.2, 0.7];
    let mut expected_output = [0.0_f32; 4];
    let mut expected_weights = [0.0_f32; 4];
    scaled_dot_product_attention_into(
        &ArrayView::new(layout, &query_host),
        &ArrayView::new(layout, &key_host),
        &ArrayView::new(layout, &value_host),
        LetoAttentionMask::Unmasked,
        0.75,
        &mut ArrayViewMut::new(layout, &mut expected_output),
        &mut ArrayViewMut::new(layout, &mut expected_weights),
    )
    .expect("Leto forward oracle");
    let mut expected_query = [1.0_f32; 4];
    let mut expected_key = [-0.5_f32; 4];
    let mut expected_value = [2.0_f32; 4];
    scaled_dot_product_attention_backward_accumulate(
        &ArrayView::new(layout, &grad_output_host),
        &ArrayView::new(layout, &query_host),
        &ArrayView::new(layout, &key_host),
        &ArrayView::new(layout, &value_host),
        &ArrayView::new(layout, &expected_weights),
        0.75,
        AttentionGradients::new(
            Some(ArrayViewMut::new(layout, &mut expected_query)),
            Some(ArrayViewMut::new(layout, &mut expected_key)),
            Some(ArrayViewMut::new(layout, &mut expected_value)),
        ),
    )
    .expect("Leto backward oracle");

    let query = device.upload(&query_host).expect("query upload");
    let key = device.upload(&key_host).expect("key upload");
    let value = device.upload(&value_host).expect("value upload");
    let grad_output = device
        .upload(&grad_output_host)
        .expect("gradient output upload");
    let output = device.alloc_zeroed::<f32>(4).expect("output allocation");
    let weights = device.alloc_zeroed::<f32>(4).expect("weights allocation");
    WgpuAttentionOps
        .attention_forward_into(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &layout),
                key: StridedView::new(&key, &layout),
                value: StridedView::new(&value, &layout),
                mask: AttentionMask::unrestricted(),
                scale: 0.75,
                output: StridedView::new(&output, &layout),
                weights: StridedView::new(&weights, &layout),
            },
        )
        .expect("WGPU forward fixture");
    let grad_query = device.upload(&[1.0_f32; 4]).expect("query gradient upload");
    let grad_key = device.upload(&[-0.5_f32; 4]).expect("key gradient upload");
    let grad_value = device.upload(&[2.0_f32; 4]).expect("value gradient upload");
    WgpuAttentionOps
        .attention_backward_accumulate(
            &device,
            AttentionBackwardOperands {
                grad_output: StridedView::new(&grad_output, &layout),
                query: StridedView::new(&query, &layout),
                key: StridedView::new(&key, &layout),
                value: StridedView::new(&value, &layout),
                weights: StridedView::new(&weights, &layout),
                scale: 0.75,
                gradients: AttentionGradientViews {
                    query: Some(StridedView::new(&grad_query, &layout)),
                    key: Some(StridedView::new(&grad_key, &layout)),
                    value: Some(StridedView::new(&grad_value, &layout)),
                },
            },
        )
        .expect("WGPU additive backward");

    assert_download_close(&device, &grad_query, &expected_query);
    assert_download_close(&device, &grad_key, &expected_key);
    assert_download_close(&device, &grad_value, &expected_value);
}

fn assert_download_close(
    device: &crate::WgpuDevice,
    buffer: &crate::WgpuBuffer<f32>,
    expected: &[f32],
) {
    let mut actual = vec![0.0_f32; expected.len()];
    device
        .download(buffer, &mut actual)
        .expect("gradient download");
    assert_close(&actual, expected, 256);
}
