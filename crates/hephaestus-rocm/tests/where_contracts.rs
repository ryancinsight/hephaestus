//! ROCm instantiation of the shared conditional-selection conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{assert_where_contract, assert_where_special_cond_values};
use hephaestus_rocm::{RocmDevice, RocmWhereOps};

#[test]
fn rocm_satisfies_the_where_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm where conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm where conformance requires a physical device: {error}"),
    };
    assert_where_contract::<_, _, i32>(&device, &RocmWhereOps);
    assert_where_contract::<_, _, f32>(&device, &RocmWhereOps);
    assert_where_special_cond_values(&device, &RocmWhereOps);
}
