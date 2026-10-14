//! ROCm instantiation of the shared concatenation conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_cat_contract;
use hephaestus_rocm::{RocmCatOps, RocmDevice};

#[test]
fn rocm_satisfies_the_cat_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm cat conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm cat conformance requires a physical device: {error}"),
    };
    assert_cat_contract::<_, _, i32>(&device, &RocmCatOps);
    assert_cat_contract::<_, _, f32>(&device, &RocmCatOps);
}
