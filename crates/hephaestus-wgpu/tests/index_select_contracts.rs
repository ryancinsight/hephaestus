//! WGPU instantiation of the shared index-selection conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::IndexSelectOps`]; this file only supplies the device
//! and the backend's seam value.

use hephaestus_conformance::assert_index_select_contract;
use hephaestus_wgpu::WgpuIndexSelectOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_index_select_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_index_select_contract::<_, _, i32>(&device, &WgpuIndexSelectOps);
    assert_index_select_contract::<_, _, f32>(&device, &WgpuIndexSelectOps);
}
