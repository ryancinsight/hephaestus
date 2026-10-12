//! ROCm instantiation of the shared inner-product conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{assert_dot_contract, assert_dot_special_float_values};
use hephaestus_rocm::{RocmDevice, RocmDotOps};

#[test]
fn rocm_satisfies_the_dot_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm dot conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm dot conformance requires a physical device: {error}"),
    };
    assert_dot_contract::<_, _, i32>(&device, &RocmDotOps);
    assert_dot_contract::<_, _, f32>(&device, &RocmDotOps);
    assert_dot_contract::<_, _, f64>(&device, &RocmDotOps);
    assert_dot_special_float_values(&device, &RocmDotOps);
}
