//! WGPU instantiation of the shared circular-shift conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::RollOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_roll_contract;
use hephaestus_wgpu::WgpuRollOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_roll_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_roll_contract::<_, _, i32>(&device, &WgpuRollOps);
    assert_roll_contract::<_, _, f32>(&device, &WgpuRollOps);
}
