//! ROCm instantiation of the shared stacking conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_stack_contract;
use hephaestus_rocm::{RocmDevice, RocmStackOps};

#[test]
fn rocm_satisfies_the_stack_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm stack conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm stack conformance requires a physical device: {error}"),
    };
    assert_stack_contract::<_, _, i32>(&device, &RocmStackOps);
    assert_stack_contract::<_, _, f32>(&device, &RocmStackOps);
}
