//! ROCm instantiation of the shared dense-product conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_dense_product_contract;
use hephaestus_rocm::{RocmDenseProductOps, RocmDevice};

#[test]
fn rocm_satisfies_the_dense_product_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm dense-product conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm dense-product conformance requires a physical device: {error}"),
    };
    assert_dense_product_contract(&device, &RocmDenseProductOps);
}
