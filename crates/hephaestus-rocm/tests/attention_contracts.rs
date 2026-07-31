//! ROCm instantiation of the shared scaled dot-product attention clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_attention_contract;
use hephaestus_rocm::{RocmAttentionOps, RocmDevice};

#[test]
fn rocm_satisfies_the_attention_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm attention conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm attention conformance requires a physical device: {error}"),
    };
    assert_attention_contract(&device, &RocmAttentionOps);
}
