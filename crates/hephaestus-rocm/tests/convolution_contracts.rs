//! ROCm instantiation of the shared convolution conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_convolution_contract;
use hephaestus_rocm::{RocmConvolutionOps, RocmDevice};

#[test]
fn rocm_satisfies_the_convolution_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm convolution conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm convolution conformance requires a physical device: {error}"),
    };
    assert_convolution_contract::<_, _, f32>(&device, &RocmConvolutionOps);
    assert_convolution_contract::<_, _, f64>(&device, &RocmConvolutionOps);
}
