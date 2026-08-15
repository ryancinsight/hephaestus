//! CUDA instantiation of scaled dot-product attention contracts.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_attention_contract;
use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionGradientViews, AttentionMask,
    AttentionOps, ComputeDevice, StridedView,
};
use hephaestus_cuda::{CudaAttentionOps, CudaDevice};
use leto::Layout;

fn device(clause: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA attention {clause}: device unavailable ({error})");
            None
        }
        Err(error) => panic!("CUDA attention {clause} requires a physical device: {error}"),
    }
}

#[test]
fn cuda_satisfies_the_attention_contract() {
    let Some(device) = device("shared contract") else {
        return;
    };
    assert_attention_contract(&device, &CudaAttentionOps);
}

#[test]
fn native_double_precision_supports_strided_views() {
    let Some(device) = device("native f64 strided contract") else {
        return;
    };
    let layout = Layout::try_new([1, 2, 2], [6, 3, 1], 1).expect("valid test layout");
    let query = device
        .upload(&[91.0_f64, 0.0, 0.0, 92.0, 0.0, 0.0])
        .expect("query upload");
    let key = device
        .upload(&[81.0_f64, 0.0, 0.0, 82.0, 0.0, 0.0])
        .expect("key upload");
    let value = device
        .upload(&[71.0_f64, 2.0, 4.0, 72.0, 6.0, 10.0])
        .expect("value upload");
    let output = device.upload(&[-3.0_f64; 6]).expect("output upload");
    let weights = device.upload(&[-3.0_f64; 6]).expect("weights upload");

    CudaAttentionOps
        .attention_forward_into(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &layout),
                key: StridedView::new(&key, &layout),
                value: StridedView::new(&value, &layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &layout),
                weights: StridedView::new(&weights, &layout),
            },
        )
        .expect("native f64 attention dispatch");

    let mut actual_output = [0.0_f64; 6];
    let mut actual_weights = [0.0_f64; 6];
    device
        .download(&output, &mut actual_output)
        .expect("output download");
    device
        .download(&weights, &mut actual_weights)
        .expect("weights download");
    assert_eq!(actual_output, [-3.0, 4.0, 7.0, -3.0, 4.0, 7.0]);
    assert_eq!(actual_weights, [-3.0, 0.5, 0.5, -3.0, 0.5, 0.5]);

    let grad_output = device
        .upload(&[61.0_f64, 1.0, 2.0, 62.0, 3.0, 4.0])
        .expect("output-gradient upload");
    let query_gradient = device
        .upload(&[-31.0_f64, 5.0, 5.0, -32.0, 5.0, 5.0])
        .expect("query-gradient upload");
    let key_gradient = device
        .upload(&[-33.0_f64, 5.0, 5.0, -34.0, 5.0, 5.0])
        .expect("key-gradient upload");
    let value_gradient = device
        .upload(&[-35.0_f64, 5.0, 5.0, -36.0, 5.0, 5.0])
        .expect("value-gradient upload");

    CudaAttentionOps
        .attention_backward_accumulate(
            &device,
            AttentionBackwardOperands {
                grad_output: StridedView::new(&grad_output, &layout),
                query: StridedView::new(&query, &layout),
                key: StridedView::new(&key, &layout),
                value: StridedView::new(&value, &layout),
                weights: StridedView::new(&weights, &layout),
                scale: 1.0,
                gradients: AttentionGradientViews {
                    query: Some(StridedView::new(&query_gradient, &layout)),
                    key: Some(StridedView::new(&key_gradient, &layout)),
                    value: Some(StridedView::new(&value_gradient, &layout)),
                },
            },
        )
        .expect("native f64 attention backward dispatch");

    let mut actual_query_gradient = [0.0_f64; 6];
    let mut actual_key_gradient = [0.0_f64; 6];
    let mut actual_value_gradient = [0.0_f64; 6];
    device
        .download(&query_gradient, &mut actual_query_gradient)
        .expect("query-gradient download");
    device
        .download(&key_gradient, &mut actual_key_gradient)
        .expect("key-gradient download");
    device
        .download(&value_gradient, &mut actual_value_gradient)
        .expect("value-gradient download");
    assert_eq!(actual_query_gradient, [-31.0, 5.0, 5.0, -32.0, 5.0, 5.0]);
    assert_eq!(actual_key_gradient, [-33.0, 5.0, 5.0, -34.0, 5.0, 5.0]);
    assert_eq!(actual_value_gradient, [-35.0, 7.0, 8.0, -36.0, 7.0, 8.0]);
}

#[test]
fn repeated_prepared_dispatch_resets_semantic_status() {
    let Some(device) = device("prepared status reset") else {
        return;
    };
    let layout = Layout::c_contiguous([1, 1, 1]).expect("scalar layout");
    let query = device.upload(&[f32::NAN]).expect("query upload");
    let finite = device.upload(&[1.0_f32]).expect("finite upload");
    let output = device.upload(&[7.0_f32]).expect("output upload");
    let weights = device.upload(&[8.0_f32]).expect("weights upload");
    let prepared = CudaAttentionOps
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
        .expect("prepared attention");

    let error = CudaAttentionOps
        .dispatch_attention_forward(&device, &prepared)
        .expect_err("non-finite query must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention query contains a non-finite value"
    );
    device
        .write_sub_buffer(&query, 0, &[1.0])
        .expect("repair query");
    CudaAttentionOps
        .dispatch_attention_forward(&device, &prepared)
        .expect("repeated prepared dispatch");

    let mut actual_output = [0.0_f32];
    let mut actual_weights = [0.0_f32];
    device
        .download(&output, &mut actual_output)
        .expect("output download");
    device
        .download(&weights, &mut actual_weights)
        .expect("weights download");
    assert_eq!(actual_output, [1.0]);
    assert_eq!(actual_weights, [1.0]);
}

#[test]
fn convex_forward_preserves_finite_extreme_values() {
    let Some(device) = device("stable convex forward") else {
        return;
    };
    let query_layout = Layout::c_contiguous([1, 1, 1]).expect("query layout");
    let key_layout = Layout::c_contiguous([1, 3, 1]).expect("key layout");
    let output_layout = Layout::c_contiguous([1, 1, 1]).expect("output layout");
    let weights_layout = Layout::c_contiguous([1, 1, 3]).expect("weights layout");
    let query = device.upload(&[0.0_f32]).expect("query upload");
    let key = device.upload(&[0.0_f32; 3]).expect("key upload");
    let value = device.upload(&[f32::MAX; 3]).expect("value upload");
    let output = device.upload(&[0.0_f32]).expect("output upload");
    let weights = device.upload(&[0.0_f32; 3]).expect("weights upload");

    CudaAttentionOps
        .attention_forward_into(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &query_layout),
                key: StridedView::new(&key, &key_layout),
                value: StridedView::new(&value, &key_layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &output_layout),
                weights: StridedView::new(&weights, &weights_layout),
            },
        )
        .expect("stable convex attention");

    let mut actual = [0.0_f32];
    device
        .download(&output, &mut actual)
        .expect("output download");
    assert_eq!(actual, [f32::MAX]);
}
