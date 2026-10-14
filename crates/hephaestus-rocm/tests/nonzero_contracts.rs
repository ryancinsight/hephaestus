//! ROCm instantiation of the shared nonzero-coordinate conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{
    assert_nonzero_contract, assert_nonzero_rejects_aliased_output,
    assert_nonzero_special_float_values,
};
use hephaestus_rocm::{RocmDevice, RocmNonzeroOps};

#[test]
fn rocm_satisfies_the_nonzero_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm nonzero conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm nonzero conformance requires a physical device: {error}"),
    };
    assert_nonzero_contract::<_, _, i32>(&device, &RocmNonzeroOps);
    assert_nonzero_contract::<_, _, f32>(&device, &RocmNonzeroOps);
    assert_nonzero_special_float_values(&device, &RocmNonzeroOps);
    assert_nonzero_rejects_aliased_output(&device, &RocmNonzeroOps);
}
