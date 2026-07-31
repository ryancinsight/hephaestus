//! WGPU instantiation of the shared ray-integral conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::RayIntegralOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_ray_integral_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuRayIntegralOps};

#[test]
fn wgpu_satisfies_the_ray_integral_contract() {
    let device = match WgpuDevice::try_default("hephaestus-ray-integral-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU ray-integral conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_ray_integral_contract(&device, &WgpuRayIntegralOps);
}
