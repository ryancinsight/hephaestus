//! ROCm instantiation of the shared CTC clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_ctc_contract;
use hephaestus_rocm::{RocmCtcOps, RocmDevice};

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
fn rocm_satisfies_the_ctc_contract() {
    let Some(device) = device("ROCm CTC conformance") else {
        return;
    };
    assert_ctc_contract(&device, &RocmCtcOps);
}
