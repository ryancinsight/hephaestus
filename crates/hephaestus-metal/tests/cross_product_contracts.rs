//! Metal instantiation of the shared cross-product conformance clause.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_cross_product_contract;
use hephaestus_metal::{MetalCrossProductOps, MetalDevice};

#[test]
fn metal_satisfies_the_cross_product_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal cross-product conformance requires a physical device");
    assert_cross_product_contract(&device, &MetalCrossProductOps);
}
