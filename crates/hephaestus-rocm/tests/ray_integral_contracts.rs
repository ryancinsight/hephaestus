//! ROCm instantiation of the shared ray-integral conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::RayIntegralOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_ray_integral_contract;
use hephaestus_rocm::{RocmDevice, RocmRayIntegralOps};

#[test]
fn rocm_satisfies_the_ray_integral_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm ray-integral conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm ray-integral conformance requires a physical device: {error}"),
    };
    assert_ray_integral_contract(&device, &RocmRayIntegralOps);
}
