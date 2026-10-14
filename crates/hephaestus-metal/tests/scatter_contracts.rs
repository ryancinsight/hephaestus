//! Metal instantiation of the shared scatter-add conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{assert_scatter_contract, assert_scatter_order_contract};
use hephaestus_metal::{MetalDevice, MetalScatterOps};

#[test]
fn metal_satisfies_the_scatter_contract() {
    let device =
        MetalDevice::try_default().expect("Metal scatter conformance requires a physical device");
    assert_scatter_contract::<_, _, i32>(&device, &MetalScatterOps::default());
    assert_scatter_contract::<_, _, f32>(&device, &MetalScatterOps::default());
    assert_scatter_order_contract(&device, &MetalScatterOps::default());
}
