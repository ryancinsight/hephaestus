//! ROCm instantiation of the shared scatter-add conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{assert_scatter_contract, assert_scatter_order_contract};
use hephaestus_rocm::{RocmDevice, RocmScatterOps};

#[test]
fn rocm_satisfies_the_scatter_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm scatter conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm scatter conformance requires a physical device: {error}"),
    };
    assert_scatter_contract::<_, _, i32>(&device, &RocmScatterOps);
    assert_scatter_contract::<_, _, f32>(&device, &RocmScatterOps);
    assert_scatter_order_contract(&device, &RocmScatterOps);
}
