//! CUDA instantiation of the shared dense-vector conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_dense_vector_contract;
use hephaestus_cuda::{CudaDevice, CudaVectorOps};

#[test]
fn cuda_satisfies_the_dense_vector_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA dense-vector conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA dense-vector conformance requires a physical device: {error}"),
    };
    let ops = CudaVectorOps::new(&device).expect("vector kernels register");
    assert_dense_vector_contract(&device, &ops);
}
