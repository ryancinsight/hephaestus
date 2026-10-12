//! Metal instantiation of the shared index-selection conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_index_select_contract;
use hephaestus_metal::{MetalDevice, MetalIndexSelectOps};

#[test]
fn metal_satisfies_the_index_select_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal index_select conformance requires a physical device");
    assert_index_select_contract::<_, _, i32>(&device, &MetalIndexSelectOps);
    assert_index_select_contract::<_, _, f32>(&device, &MetalIndexSelectOps);
}
