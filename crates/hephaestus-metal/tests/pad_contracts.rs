//! Metal instantiation of the shared pad conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_pad_contract;
use hephaestus_metal::{MetalDevice, MetalPadOps};

#[test]
fn metal_satisfies_the_pad_contract() {
    let device =
        MetalDevice::try_default().expect("Metal pad conformance requires a physical device");
    assert_pad_contract::<_, _, i32>(&device, &MetalPadOps);
    assert_pad_contract::<_, _, f32>(&device, &MetalPadOps);
}
