//! Metal instantiation of the shared conditional-selection conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{assert_where_contract, assert_where_special_cond_values};
use hephaestus_metal::{MetalDevice, MetalWhereOps};

#[test]
fn metal_satisfies_the_where_contract() {
    let device =
        MetalDevice::try_default().expect("Metal where conformance requires a physical device");
    assert_where_contract::<_, _, i32>(&device, &MetalWhereOps::default());
    assert_where_contract::<_, _, f32>(&device, &MetalWhereOps::default());
    assert_where_special_cond_values(&device, &MetalWhereOps::default());
}
