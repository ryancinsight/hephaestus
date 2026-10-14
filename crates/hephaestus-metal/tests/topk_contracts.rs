//! Metal instantiation of the shared top-k conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{assert_topk_contract, assert_topk_strided_output_contract};
use hephaestus_metal::{MetalDevice, MetalTopKOps};

#[test]
fn metal_satisfies_the_topk_contract() {
    let device =
        MetalDevice::try_default().expect("Metal top-k conformance requires a physical device");
    assert_topk_contract(&device, &MetalTopKOps::default());
    assert_topk_strided_output_contract(&device, &MetalTopKOps::default());
}
