//! Metal instantiation of the shared dense-product conformance clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::{
    assert_dense_composition_contract, assert_dense_matrix_function_contract,
    assert_dense_product_contract,
};
use hephaestus_metal::{MetalDenseProductOps, MetalDevice};

#[test]
fn metal_satisfies_the_dense_product_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal dense-product conformance requires a physical device");
    let ops = MetalDenseProductOps::default();
    assert_dense_product_contract(&device, &ops);
    assert_dense_composition_contract(&device, &ops);
    assert_dense_matrix_function_contract(&device, &ops);
}
