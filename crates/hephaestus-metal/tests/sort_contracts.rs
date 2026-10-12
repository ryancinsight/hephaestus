//! Metal instantiation of the shared stable-sort conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{assert_sort_contract, assert_sort_special_float_values};
use hephaestus_metal::{MetalDevice, MetalSortOps};

#[test]
fn metal_satisfies_the_sort_contract() {
    let device =
        MetalDevice::try_default().expect("Metal sort conformance requires a physical device");
    assert_sort_contract::<_, _, i32>(&device, &MetalSortOps);
    assert_sort_contract::<_, _, f32>(&device, &MetalSortOps);
    assert_sort_special_float_values(&device, &MetalSortOps);
}
