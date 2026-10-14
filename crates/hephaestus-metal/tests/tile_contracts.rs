//! Metal instantiation of the shared tiling conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_tile_contract;
use hephaestus_metal::{MetalDevice, MetalTileOps};

#[test]
fn metal_satisfies_the_tile_contract() {
    let device =
        MetalDevice::try_default().expect("Metal tile conformance requires a physical device");
    assert_tile_contract::<_, _, i32>(&device, &MetalTileOps::default());
    assert_tile_contract::<_, _, f32>(&device, &MetalTileOps::default());
}
