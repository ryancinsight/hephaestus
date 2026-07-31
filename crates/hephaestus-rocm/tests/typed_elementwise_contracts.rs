//! ROCm instantiation of the shared typed-elementwise conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value. The
//! clauses cover the scalar-aware comparison dispatch paths, four of the six
//! shared entry points no backend exercised before `ATLAS-ARCH-001`.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_typed_elementwise_contract;
use hephaestus_rocm::{RocmDevice, RocmElementwiseOps};

#[test]
fn rocm_satisfies_the_typed_elementwise_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm typed-elementwise conformance: device unavailable ({error})");
            return;
        }
        Err(error) => {
            panic!("ROCm typed-elementwise conformance requires a physical device: {error}")
        }
    };
    assert_typed_elementwise_contract(&device, &RocmElementwiseOps);
}
