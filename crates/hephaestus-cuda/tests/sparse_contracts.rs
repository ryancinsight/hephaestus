//! CUDA instantiation of the shared sparse-operator conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::SparseOperatorOps`]; this file only supplies the device
//! and the backend's seam value.

#![cfg(feature = "cuda")]

use hephaestus_conformance::{assert_batch_submit_contract, assert_sparse_operator_contract};
use hephaestus_cuda::{CudaDevice, CudaSparseOps};

#[test]
fn cuda_satisfies_the_sparse_operator_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA sparse conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA sparse conformance requires a physical device: {error}"),
    };
    assert_sparse_operator_contract(&device, &CudaSparseOps);
    assert_batch_submit_contract(&device, &CudaSparseOps);
}
