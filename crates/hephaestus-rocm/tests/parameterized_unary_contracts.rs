//! ROCm instantiation of the shared runtime-parameter unary contract.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_parameterized_unary_contract;
use hephaestus_rocm::{RocmDevice, RocmParameterizedUnaryOps};

#[test]
fn rocm_satisfies_the_parameterized_unary_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm parameterized-unary conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm parameterized-unary conformance requires a device: {error}"),
    };
    assert_parameterized_unary_contract(&device, &RocmParameterizedUnaryOps);
}
