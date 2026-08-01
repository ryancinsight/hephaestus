//! ROCm instantiation of the shared transfer conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_transfer_contract;
use hephaestus_rocm::RocmDevice;

#[test]
fn rocm_satisfies_the_transfer_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm transfer conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm transfer conformance requires a physical device: {error}"),
    };
    assert_transfer_contract(&device);
}
