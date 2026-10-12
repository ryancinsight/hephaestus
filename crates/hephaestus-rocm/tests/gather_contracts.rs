//! ROCm instantiation of the shared gather conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_gather_contract;
use hephaestus_rocm::{RocmDevice, RocmGatherOps};

#[test]
fn rocm_satisfies_the_gather_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm gather conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm gather conformance requires a physical device: {error}"),
    };
    assert_gather_contract::<_, _, i32>(&device, &RocmGatherOps);
    assert_gather_contract::<_, _, f32>(&device, &RocmGatherOps);
}
