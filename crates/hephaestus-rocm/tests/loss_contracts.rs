//! ROCm instantiation of provider-owned mean cross-entropy contracts.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_cross_entropy_contract;
use hephaestus_rocm::{RocmCrossEntropyOps, RocmDevice};

fn device(clause: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip {clause}: ROCm device unavailable ({error})");
            None
        }
        Err(error) => panic!("{clause} requires a physical ROCm device: {error}"),
    }
}

#[test]
fn rocm_satisfies_shared_cross_entropy_contract() {
    let Some(device) = device("ROCm cross-entropy conformance") else {
        return;
    };
    assert_cross_entropy_contract(&device, &RocmCrossEntropyOps);
}
