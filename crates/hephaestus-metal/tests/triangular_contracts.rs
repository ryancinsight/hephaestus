//! Metal instantiation of the shared triangular conformance clause.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_triangular_contract;
use hephaestus_metal::{MetalDevice, MetalTriangularOps};

#[test]
fn metal_satisfies_the_triangular_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal triangular conformance requires a physical device");
    assert_triangular_contract(&device, &MetalTriangularOps::default());
}
