//! Metal instantiation of the shared scan conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ScanOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_scan_contract;
use hephaestus_metal::{MetalDevice, MetalScanOps};

#[test]
fn metal_satisfies_the_scan_contract() {
    let device =
        MetalDevice::try_default().expect("Metal scan conformance requires a physical device");
    assert_scan_contract(&device, &MetalScanOps::default());
}
