//! ROCm instantiation of the shared random-initialization clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_random_init_contract;
use hephaestus_rocm::{RocmDevice, RocmRandomOps};

#[test]
fn rocm_satisfies_the_random_init_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm random conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm random conformance requires a physical device: {error}"),
    };
    assert_random_init_contract(&device, &RocmRandomOps);
}
