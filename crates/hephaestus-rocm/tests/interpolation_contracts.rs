//! ROCm instantiation of the shared interpolation conformance clause.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_interpolation_contract;
use hephaestus_rocm::{RocmDevice, RocmInterpolationOps};

#[test]
fn rocm_satisfies_the_interpolation_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm interpolation conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm interpolation conformance requires a physical device: {error}"),
    };
    assert_interpolation_contract(&device, &RocmInterpolationOps);
}
