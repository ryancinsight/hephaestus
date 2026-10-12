//! ROCm instantiation of the shared top-k conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{assert_topk_contract, assert_topk_strided_output_contract};
use hephaestus_rocm::{RocmDevice, RocmTopKOps};

#[test]
fn rocm_satisfies_the_topk_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm top-k conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm top-k conformance requires a physical device: {error}"),
    };
    assert_topk_contract(&device, &RocmTopKOps);
    assert_topk_strided_output_contract(&device, &RocmTopKOps);
}
