//! ROCm instantiation of the shared scan conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ScanOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_scan_contract;
use hephaestus_rocm::{RocmDevice, RocmScanOps};

#[test]
fn rocm_satisfies_the_scan_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm scan conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm scan conformance requires a physical device: {error}"),
    };
    assert_scan_contract(&device, &RocmScanOps);
}
