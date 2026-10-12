//! Metal instantiation of the shared splitting conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_split_contract;
use hephaestus_metal::{MetalDevice, MetalSplitOps};

#[test]
fn metal_satisfies_the_split_contract() {
    let device =
        MetalDevice::try_default().expect("Metal split conformance requires a physical device");
    assert_split_contract::<_, _, i32>(&device, &MetalSplitOps);
    assert_split_contract::<_, _, f32>(&device, &MetalSplitOps);
}
