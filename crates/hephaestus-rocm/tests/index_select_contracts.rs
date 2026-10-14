//! ROCm instantiation of the shared index-selection conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_index_select_contract;
use hephaestus_rocm::{RocmDevice, RocmIndexSelectOps};

#[test]
fn rocm_satisfies_the_index_select_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm index_select conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm index_select conformance requires a physical device: {error}"),
    };
    assert_index_select_contract::<_, _, i32>(&device, &RocmIndexSelectOps);
    assert_index_select_contract::<_, _, f32>(&device, &RocmIndexSelectOps);
}
