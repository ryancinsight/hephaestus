//! CUDA instantiation of the shared decomposition conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::DecompositionOps`]; this file only supplies the device
//! and the backend seam value.

#![cfg(all(feature = "cuda", feature = "decomposition"))]
use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_cuda::{CudaDecompositionOps, CudaDevice};

#[test]
fn cuda_satisfies_the_decomposition_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA decomposition conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA decomposition conformance requires a physical device: {error}"),
    };
    assert_decomposition_contract(&device, &CudaDecompositionOps);
}
