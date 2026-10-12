//! ROCm instantiation of the shared triangular conformance clause.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_triangular_contract;
use hephaestus_rocm::{RocmDevice, RocmTriangularOps};

#[test]
fn rocm_satisfies_the_triangular_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm triangular conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm triangular conformance requires a physical device: {error}"),
    };
    assert_triangular_contract(&device, &RocmTriangularOps);
}
