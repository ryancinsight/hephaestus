//! ROCm instantiation of the shared decomposition conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::DecompositionOps`]; this file only supplies the device
//! and the backend seam value.

#![cfg(all(feature = "rocm", feature = "decomposition", target_os = "linux"))]
use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_rocm::{RocmDecompositionOps, RocmDevice};

#[test]
fn rocm_satisfies_the_decomposition_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm decomposition conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm decomposition conformance requires a physical device: {error}"),
    };
    assert_decomposition_contract(&device, &RocmDecompositionOps);
}
