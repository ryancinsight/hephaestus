//! ROCm instantiation of the shared masked-fill conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{assert_masked_fill_contract, assert_masked_fill_special_mask_values};
use hephaestus_rocm::{RocmDevice, RocmMaskedFillOps};

#[test]
fn rocm_satisfies_the_masked_fill_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm masked_fill conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm masked_fill conformance requires a physical device: {error}"),
    };
    assert_masked_fill_contract::<_, _, i32>(&device, &RocmMaskedFillOps);
    assert_masked_fill_contract::<_, _, f32>(&device, &RocmMaskedFillOps);
    assert_masked_fill_special_mask_values(&device, &RocmMaskedFillOps);
}
