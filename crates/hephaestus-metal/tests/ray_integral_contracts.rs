//! Metal instantiation of the shared ray-integral conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::RayIntegralOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_ray_integral_contract;
use hephaestus_metal::{MetalDevice, MetalRayIntegralOps};

#[test]
fn metal_satisfies_the_ray_integral_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal ray-integral conformance requires a physical device");
    assert_ray_integral_contract(&device, &MetalRayIntegralOps);
}
