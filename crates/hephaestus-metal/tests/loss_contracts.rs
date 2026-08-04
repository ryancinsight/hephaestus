//! Metal instantiation of provider-owned mean cross-entropy contracts.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_cross_entropy_contract;
use hephaestus_metal::{MetalCrossEntropyOps, MetalDevice};

#[test]
fn metal_satisfies_shared_cross_entropy_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal cross-entropy conformance requires a physical device");
    assert_cross_entropy_contract(&device, &MetalCrossEntropyOps);
}
