//! WGPU instantiation of the shared scan conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ScanOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_scan_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuScanOps};

#[test]
fn wgpu_satisfies_the_scan_contract() {
    let device = match WgpuDevice::try_default("hephaestus-scan-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU scan conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_scan_contract(&device, &WgpuScanOps);
}
