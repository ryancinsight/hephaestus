//! ROCm instantiation of the shared adaptive-pooling conformance clause.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_adaptive_pooling_contract;
use hephaestus_rocm::{RocmAdaptivePoolingOps, RocmDevice};

#[test]
fn rocm_satisfies_the_adaptive_pooling_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm adaptive-pooling conformance: device unavailable ({error})");
            return;
        }
        Err(error) => {
            panic!("ROCm adaptive-pooling conformance requires a physical device: {error}")
        }
    };
    assert_adaptive_pooling_contract(&device, &RocmAdaptivePoolingOps);
}
