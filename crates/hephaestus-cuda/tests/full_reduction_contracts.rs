//! CUDA instantiation of the shared full-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::FullReductionOps`]; this file only supplies the device
//! and the backend's seam value.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_full_reduction_contract;
use hephaestus_cuda::{CudaDevice, CudaFullReductionOps};

#[test]
fn cuda_satisfies_the_full_reduction_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA full-reduction conformance: device unavailable ({error})");
            return;
        }
        Err(error) => {
            panic!("CUDA full-reduction conformance requires a physical device: {error}")
        }
    };
    assert_full_reduction_contract(&device, &CudaFullReductionOps);
}
