//! WGPU instantiation of the shared conditional-selection conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::WhereOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::{assert_where_contract, assert_where_special_cond_values};
use hephaestus_wgpu::WgpuWhereOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_where_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_where_contract::<_, _, i32>(&device, &WgpuWhereOps);
    assert_where_contract::<_, _, f32>(&device, &WgpuWhereOps);
    assert_where_special_cond_values(&device, &WgpuWhereOps);
}
