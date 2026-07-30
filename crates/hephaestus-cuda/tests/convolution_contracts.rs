//! CUDA instantiation of the shared convolution conformance clauses.

#![cfg(feature = "cuda")]

use hephaestus_conformance::{assert_convolution_contract, assert_convolution_f64_contract};
use hephaestus_cuda::{CudaConvolutionOps, CudaDevice};

#[test]
fn cuda_satisfies_the_convolution_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA convolution conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA convolution conformance requires a physical device: {error}"),
    };
    assert_convolution_contract(&device, &CudaConvolutionOps);
    assert_convolution_f64_contract(&device, &CudaConvolutionOps);
}
