//! Metal instantiation of the shared nonzero-coordinate conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{
    assert_nonzero_contract, assert_nonzero_rejects_aliased_output,
    assert_nonzero_special_float_values,
};
use hephaestus_metal::{MetalDevice, MetalNonzeroOps};

#[test]
fn metal_satisfies_the_nonzero_contract() {
    let device =
        MetalDevice::try_default().expect("Metal nonzero conformance requires a physical device");
    assert_nonzero_contract::<_, _, i32>(&device, &MetalNonzeroOps::default());
    assert_nonzero_contract::<_, _, f32>(&device, &MetalNonzeroOps::default());
    assert_nonzero_special_float_values(&device, &MetalNonzeroOps::default());
    assert_nonzero_rejects_aliased_output(&device, &MetalNonzeroOps::default());
}
