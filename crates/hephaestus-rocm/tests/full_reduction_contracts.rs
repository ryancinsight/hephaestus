//! ROCm instantiation of the shared full-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::FullReductionOps`]; this file only supplies the device
//! and the backend's seam value.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_full_reduction_contract;
use hephaestus_rocm::{RocmDevice, RocmFullReductionOps};

#[test]
fn rocm_satisfies_the_full_reduction_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm full-reduction conformance: device unavailable ({error})");
            return;
        }
        Err(error) => {
            panic!("ROCm full-reduction conformance requires a physical device: {error}")
        }
    };
    assert_full_reduction_contract(&device, &RocmFullReductionOps);
}
