//! WGPU instantiation of the shared gather conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::GatherOps`]; this
//! file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_gather_contract;
use hephaestus_wgpu::WgpuGatherOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_gather_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_gather_contract::<_, _, i32>(&device, &WgpuGatherOps);
    assert_gather_contract::<_, _, f32>(&device, &WgpuGatherOps);
}
