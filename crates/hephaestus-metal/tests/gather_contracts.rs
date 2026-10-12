//! Metal instantiation of the shared gather conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_gather_contract;
use hephaestus_metal::{MetalDevice, MetalGatherOps};

#[test]
fn metal_satisfies_the_gather_contract() {
    let device =
        MetalDevice::try_default().expect("Metal gather conformance requires a physical device");
    assert_gather_contract::<_, _, i32>(&device, &MetalGatherOps);
    assert_gather_contract::<_, _, f32>(&device, &MetalGatherOps);
}
