//! CUDA instantiation of the shared convolution conformance clauses.

#![cfg(feature = "cuda")]

use hephaestus_conformance::{assert_convolution_contract, assert_convolution_f64_contract};
use hephaestus_cuda::{CudaConvolutionOps, CudaDevice};

#[test]
fn cuda_satisfies_the_convolution_contract() {
    let device =
        CudaDevice::try_default().expect("CUDA convolution conformance requires a physical device");
    assert_convolution_contract(&device, &CudaConvolutionOps);
    assert_convolution_f64_contract(&device, &CudaConvolutionOps);
}
