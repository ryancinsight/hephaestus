//! Metal instantiation of the shared runtime-parameter unary contract.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_parameterized_unary_contract;
use hephaestus_metal::{MetalDevice, MetalParameterizedUnaryOps};

#[test]
fn metal_satisfies_the_parameterized_unary_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal parameterized-unary conformance requires a physical device");
    assert_parameterized_unary_contract(&device, &MetalParameterizedUnaryOps);
}
