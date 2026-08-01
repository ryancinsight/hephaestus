//! Metal instantiation of the shared dense-product conformance clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_dense_product_contract;
use hephaestus_metal::{MetalDenseProductOps, MetalDevice};

#[test]
fn metal_satisfies_the_dense_product_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal dense-product conformance requires a physical device");
    assert_dense_product_contract(&device, &MetalDenseProductOps::default());
}
