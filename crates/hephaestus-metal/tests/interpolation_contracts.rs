//! Metal instantiation of the shared interpolation conformance clause.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_interpolation_contract;
use hephaestus_metal::{MetalDevice, MetalInterpolationOps};

#[test]
fn metal_satisfies_the_interpolation_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal interpolation conformance requires a physical device");
    assert_interpolation_contract(&device, &MetalInterpolationOps);
}
