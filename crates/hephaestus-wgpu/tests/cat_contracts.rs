//! WGPU instantiation of the shared concatenation conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::CatOps`]; this
//! file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_cat_contract;
use hephaestus_wgpu::WgpuCatOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_cat_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_cat_contract::<_, _, i32>(&device, &WgpuCatOps);
    assert_cat_contract::<_, _, f32>(&device, &WgpuCatOps);
}
