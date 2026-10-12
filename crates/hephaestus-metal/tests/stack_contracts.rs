//! Metal instantiation of the shared stacking conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_stack_contract;
use hephaestus_metal::{MetalDevice, MetalStackOps};

#[test]
fn metal_satisfies_the_stack_contract() {
    let device =
        MetalDevice::try_default().expect("Metal stack conformance requires a physical device");
    assert_stack_contract::<_, _, i32>(&device, &MetalStackOps);
    assert_stack_contract::<_, _, f32>(&device, &MetalStackOps);
}
