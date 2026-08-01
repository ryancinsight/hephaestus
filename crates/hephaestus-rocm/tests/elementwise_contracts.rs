//! ROCm instantiation of the shared untyped elementwise clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_elementwise_contract;
use hephaestus_rocm::{RocmDevice, RocmElementwiseOps};

#[test]
fn rocm_satisfies_the_elementwise_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm elementwise conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm elementwise conformance requires a physical device: {error}"),
    };
    assert_elementwise_contract(&device, &RocmElementwiseOps);
}
