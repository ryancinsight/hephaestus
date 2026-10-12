//! ROCm instantiation of the shared stable-sort conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{assert_sort_contract, assert_sort_special_float_values};
use hephaestus_rocm::{RocmDevice, RocmSortOps};

#[test]
fn rocm_satisfies_the_sort_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm sort conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm sort conformance requires a physical device: {error}"),
    };
    assert_sort_contract::<_, _, i32>(&device, &RocmSortOps);
    assert_sort_contract::<_, _, f32>(&device, &RocmSortOps);
    assert_sort_special_float_values(&device, &RocmSortOps);
}
