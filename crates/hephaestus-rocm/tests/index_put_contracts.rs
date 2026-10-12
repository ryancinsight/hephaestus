//! ROCm instantiation of the shared scatter-assignment conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_index_put_contract;
use hephaestus_rocm::{RocmDevice, RocmIndexPutOps};

#[test]
fn rocm_satisfies_the_index_put_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm index_put conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm index_put conformance requires a physical device: {error}"),
    };
    assert_index_put_contract::<_, _, i32>(&device, &RocmIndexPutOps);
    assert_index_put_contract::<_, _, f32>(&device, &RocmIndexPutOps);
}
