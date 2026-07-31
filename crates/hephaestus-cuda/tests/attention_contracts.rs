//! CUDA instantiation of scaled dot-product attention contracts.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_attention_contract;
use hephaestus_core::{
    AttentionForwardOperands, AttentionMask, AttentionOps, ComputeDevice, StridedView,
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
    let layout = Layout::new([1, 2, 2], [6, 3, 1], 1);
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
}
