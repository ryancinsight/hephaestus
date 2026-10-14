//! WGPU instantiation of the shared stacking conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::StackOps`]; this
//! file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_stack_contract;
use hephaestus_wgpu::WgpuStackOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_stack_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_stack_contract::<_, _, i32>(&device, &WgpuStackOps);
    assert_stack_contract::<_, _, f32>(&device, &WgpuStackOps);
}
