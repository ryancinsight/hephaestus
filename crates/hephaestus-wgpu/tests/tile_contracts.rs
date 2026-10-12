//! WGPU instantiation of the shared tiling conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::TileOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_tile_contract;
use hephaestus_wgpu::WgpuTileOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_tile_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_tile_contract::<_, _, i32>(&device, &WgpuTileOps);
    assert_tile_contract::<_, _, f32>(&device, &WgpuTileOps);
}
