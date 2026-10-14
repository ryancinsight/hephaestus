//! ROCm instantiation of the shared cross-product conformance clause.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_cross_product_contract;
use hephaestus_rocm::{RocmCrossProductOps, RocmDevice};

#[test]
fn rocm_satisfies_the_cross_product_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm cross-product conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm cross-product conformance requires a physical device: {error}"),
    };
    assert_cross_product_contract(&device, &RocmCrossProductOps);
}
