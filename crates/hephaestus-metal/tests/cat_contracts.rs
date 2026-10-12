//! Metal instantiation of the shared concatenation conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_cat_contract;
use hephaestus_metal::{MetalCatOps, MetalDevice};

#[test]
fn metal_satisfies_the_cat_contract() {
    let device =
        MetalDevice::try_default().expect("Metal cat conformance requires a physical device");
    assert_cat_contract::<_, _, i32>(&device, &MetalCatOps);
    assert_cat_contract::<_, _, f32>(&device, &MetalCatOps);
}
