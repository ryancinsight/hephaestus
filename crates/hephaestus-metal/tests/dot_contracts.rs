//! Metal instantiation of the shared inner-product conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{assert_dot_contract, assert_dot_special_float_values};
use hephaestus_metal::{MetalDevice, MetalDotOps};

#[test]
fn metal_satisfies_the_dot_contract() {
    let device =
        MetalDevice::try_default().expect("Metal dot conformance requires a physical device");
    assert_dot_contract::<_, _, i32>(&device, &MetalDotOps::default());
    assert_dot_contract::<_, _, f32>(&device, &MetalDotOps::default());
    assert_dot_special_float_values(&device, &MetalDotOps::default());
}
