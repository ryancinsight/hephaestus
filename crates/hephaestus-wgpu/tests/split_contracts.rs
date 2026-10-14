//! WGPU instantiation of the shared splitting conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::SplitOps`]; this
//! file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_split_contract;
use hephaestus_wgpu::WgpuSplitOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_split_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_split_contract::<_, _, i32>(&device, &WgpuSplitOps);
    assert_split_contract::<_, _, f32>(&device, &WgpuSplitOps);
}
