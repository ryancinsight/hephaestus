//! Metal instantiation of the shared adaptive-pooling conformance clause.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_adaptive_pooling_contract;
use hephaestus_metal::{MetalAdaptivePoolingOps, MetalDevice};

#[test]
fn metal_satisfies_the_adaptive_pooling_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal adaptive-pooling conformance requires a physical device");
    assert_adaptive_pooling_contract(&device, &MetalAdaptivePoolingOps::default());
}
