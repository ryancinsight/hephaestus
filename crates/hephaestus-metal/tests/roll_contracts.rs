//! Metal instantiation of the shared circular-shift conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_roll_contract;
use hephaestus_metal::{MetalDevice, MetalRollOps};

#[test]
fn metal_satisfies_the_roll_contract() {
    let device =
        MetalDevice::try_default().expect("Metal roll conformance requires a physical device");
    assert_roll_contract::<_, _, i32>(&device, &MetalRollOps);
    assert_roll_contract::<_, _, f32>(&device, &MetalRollOps);
}
