//! WGPU instantiation of the shared stable-sort conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::SortOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::{assert_sort_contract, assert_sort_special_float_values};
use hephaestus_wgpu::WgpuSortOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_sort_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_sort_contract::<_, _, i32>(&device, &WgpuSortOps);
    assert_sort_contract::<_, _, f32>(&device, &WgpuSortOps);
    assert_sort_special_float_values(&device, &WgpuSortOps);
}
