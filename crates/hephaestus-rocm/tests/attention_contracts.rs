//! ROCm instantiation of the shared scaled dot-product attention clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_attention_contract;
use hephaestus_core::{
    AttentionForwardOperands, AttentionMask, AttentionOps, ComputeDevice, StridedView,
};
use hephaestus_rocm::{RocmAttentionOps, RocmDevice};
use leto::Layout;

fn device(clause: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip {clause}: ROCm device unavailable ({error})");
            None
        }
        Err(error) => panic!("{clause} requires a physical ROCm device: {error}"),
    }
}

#[test]
fn rocm_satisfies_the_attention_contract() {
    let Some(device) = device("ROCm attention conformance") else {
        return;
    };
    assert_attention_contract(&device, &RocmAttentionOps);
}

#[test]
fn native_double_forward_is_exact() {
    let Some(device) = device("ROCm f64 attention") else {
        return;
    };
    let layout = Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid test layout");
    let query = device.upload(&[2.0_f64]).expect("query upload");
    let key = device.upload(&[3.0_f64]).expect("key upload");
    let value = device.upload(&[7.0_f64]).expect("value upload");
    let output = device.upload(&[-1.0_f64]).expect("output upload");
    let weights = device.upload(&[-1.0_f64]).expect("weights upload");

    RocmAttentionOps
        .attention_forward_into(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &layout),
                key: StridedView::new(&key, &layout),
                value: StridedView::new(&value, &layout),
                mask: AttentionMask::unrestricted(),
                scale: 0.5,
                output: StridedView::new(&output, &layout),
                weights: StridedView::new(&weights, &layout),
            },
        )
        .expect("native f64 attention");

    let mut output_host = [0.0_f64];
    let mut weights_host = [0.0_f64];
    device
        .download(&output, &mut output_host)
        .expect("output readback");
    device
        .download(&weights, &mut weights_host)
        .expect("weights readback");
    assert_eq!(output_host, [7.0]);
    assert_eq!(weights_host, [1.0]);
}

#[test]
fn repeated_prepared_dispatch_resets_semantic_status() {
    let Some(device) = device("ROCm repeated attention dispatch") else {
        return;
    };
    let layout = Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid test layout");
    let query = device.upload(&[f32::NAN]).expect("query upload");
    let finite = device.upload(&[1.0_f32]).expect("finite upload");
    let output = device.upload(&[7.0_f32]).expect("output upload");
    let weights = device.upload(&[8.0_f32]).expect("weights upload");
    let prepared = RocmAttentionOps
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
        .expect("attention preparation");

    let error = RocmAttentionOps
        .dispatch_attention_forward(&device, &prepared)
        .expect_err("non-finite query must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention query contains a non-finite value"
    );
    device
        .write_buffer(&query, &[1.0])
        .expect("repair query in place");
    RocmAttentionOps
        .dispatch_attention_forward(&device, &prepared)
        .expect("second prepared dispatch");

    let mut output_host = [0.0_f32];
    let mut weights_host = [0.0_f32];
    device
        .download(&output, &mut output_host)
        .expect("output readback");
    device
        .download(&weights, &mut weights_host)
        .expect("weights readback");
    assert_eq!(output_host, [1.0]);
    assert_eq!(weights_host, [1.0]);
}
