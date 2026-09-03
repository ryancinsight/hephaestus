//! WGPU instantiation of the shared scaled dot-product attention contract.

use hephaestus_conformance::assert_attention_contract;
use hephaestus_core::{
    AttentionForwardOperands, AttentionMask, AttentionOps, ComputeDevice, StridedView,
};
use hephaestus_wgpu::{WgpuAttentionOps, WgpuDevice};
use leto::Layout;

pub(super) fn wgpu_satisfies_the_attention_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_attention_contract(&device, &WgpuAttentionOps);
}

pub(super) fn prepared_dispatch_resets_semantic_status_after_failure() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let layout = Layout::c_contiguous([1, 1, 1]).expect("scalar layout");
    let query = device.upload(&[f32::NAN]).expect("query upload");
    let finite = device.upload(&[1.0_f32]).expect("finite upload");
    let output = device.upload(&[7.0_f32]).expect("output upload");
    let weights = device.upload(&[8.0_f32]).expect("weights upload");
    let prepared = WgpuAttentionOps
        .prepare_attention_forward(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &layout),
                key: StridedView::new(&finite, &layout),
                value: StridedView::new(&finite, &layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &layout),
                weights: StridedView::new(&weights, &layout),
            },
        )
        .expect("prepare reusable forward");
    let error = WgpuAttentionOps
        .dispatch_attention_forward(&device, &prepared)
        .expect_err("non-finite query must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention query contains a non-finite value"
    );

    device
        .queue()
        .write_buffer(query.raw(), 0, eunomia::layout::bytes_of(&1.0_f32));
    WgpuAttentionOps
        .dispatch_attention_forward(&device, &prepared)
        .expect("status reset permits corrected repeated dispatch");
    assert_download_eq(&device, &output, &[1.0]);
    assert_download_eq(&device, &weights, &[1.0]);
}

pub(super) fn zero_probability_prefix_preserves_stable_convex_output() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let query_layout = Layout::c_contiguous([1, 1, 1]).expect("query layout");
    let key_layout = Layout::c_contiguous([1, 2, 1]).expect("key layout");
    let weights_layout = Layout::c_contiguous([1, 1, 2]).expect("weights layout");
    let query = device.upload(&[1.0_f32]).expect("query upload");
    let key = device.upload(&[-1_000.0_f32, 0.0]).expect("key upload");
    let value = device.upload(&[f32::MAX, 2.0]).expect("value upload");
    let output = device.upload(&[7.0_f32]).expect("output upload");
    let weights = device.upload(&[8.0_f32; 2]).expect("weights upload");

    WgpuAttentionOps
        .attention_forward_into(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &query_layout),
                key: StridedView::new(&key, &key_layout),
                value: StridedView::new(&value, &key_layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &query_layout),
                weights: StridedView::new(&weights, &weights_layout),
            },
        )
        .expect("underflowed zero-probability prefix is valid");
    assert_download_eq(&device, &output, &[2.0]);
    assert_download_eq(&device, &weights, &[0.0, 1.0]);
}

fn device_or_skip() -> Option<WgpuDevice> {
    super::device_or_skip()
}

fn assert_download_eq(
    device: &WgpuDevice,
    buffer: &hephaestus_wgpu::WgpuBuffer<f32>,
    expected: &[f32],
) {
    let mut actual = vec![0.0_f32; expected.len()];
    device
        .download(buffer, &mut actual)
        .expect("download result");
    assert_eq!(actual, expected);
}
