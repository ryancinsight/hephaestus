//! WGPU instantiation of the shared scatter-add conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ScatterOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::{assert_scatter_contract, assert_scatter_order_contract};
use hephaestus_wgpu::WgpuScatterOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_scatter_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_scatter_contract::<_, _, i32>(&device, &WgpuScatterOps);
    assert_scatter_contract::<_, _, f32>(&device, &WgpuScatterOps);
    assert_scatter_order_contract(&device, &WgpuScatterOps);
}
