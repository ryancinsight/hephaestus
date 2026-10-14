//! ROCm instantiation of the shared splitting conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_split_contract;
use hephaestus_rocm::{RocmDevice, RocmSplitOps};

#[test]
fn rocm_satisfies_the_split_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm split conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm split conformance requires a physical device: {error}"),
    };
    assert_split_contract::<_, _, i32>(&device, &RocmSplitOps);
    assert_split_contract::<_, _, f32>(&device, &RocmSplitOps);
}
