//! WGPU instantiation of the shared ray-integral conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::RayIntegralOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_ray_integral_contract;
use hephaestus_wgpu::WgpuRayIntegralOps;

pub(super) fn wgpu_satisfies_the_ray_integral_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_ray_integral_contract(&device, &WgpuRayIntegralOps);
}
