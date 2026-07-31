//! CUDA instantiation of the shared runtime-parameter unary contract.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_parameterized_unary_contract;
use hephaestus_cuda::{CudaDevice, CudaParameterizedUnaryOps};

#[test]
fn cuda_satisfies_the_parameterized_unary_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA parameterized-unary conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA parameterized-unary conformance requires a device: {error}"),
    };
    assert_parameterized_unary_contract(&device, &CudaParameterizedUnaryOps);
}
