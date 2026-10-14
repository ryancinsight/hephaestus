//! ROCm instantiation of the shared circular-shift conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_roll_contract;
use hephaestus_rocm::{RocmDevice, RocmRollOps};

#[test]
fn rocm_satisfies_the_roll_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm roll conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm roll conformance requires a physical device: {error}"),
    };
    assert_roll_contract::<_, _, i32>(&device, &RocmRollOps);
    assert_roll_contract::<_, _, f32>(&device, &RocmRollOps);
}
