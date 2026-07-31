//! CUDA instantiation of the shared scan conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ScanOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_scan_contract;
use hephaestus_cuda::{CudaDevice, CudaScanOps};

#[test]
fn cuda_satisfies_the_scan_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA scan conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA scan conformance requires a physical device: {error}"),
    };
    assert_scan_contract(&device, &CudaScanOps);
}
