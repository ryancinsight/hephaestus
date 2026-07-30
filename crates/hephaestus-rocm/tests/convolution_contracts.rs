//! ROCm instantiation of the shared convolution conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::{assert_convolution_contract, assert_convolution_f64_contract};
use hephaestus_rocm::{RocmConvolutionOps, RocmDevice};

#[test]
fn rocm_satisfies_the_convolution_contract() {
    let device =
        RocmDevice::try_default().expect("ROCm convolution conformance requires a physical device");
    assert_convolution_contract(&device, &RocmConvolutionOps);
    assert_convolution_f64_contract(&device, &RocmConvolutionOps);
}
