//! ROCm instantiation of the shared pad conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_pad_contract;
use hephaestus_rocm::{RocmDevice, RocmPadOps};

#[test]
fn rocm_satisfies_the_pad_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm pad conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm pad conformance requires a physical device: {error}"),
    };
    assert_pad_contract::<_, _, i32>(&device, &RocmPadOps);
    assert_pad_contract::<_, _, f32>(&device, &RocmPadOps);
}
