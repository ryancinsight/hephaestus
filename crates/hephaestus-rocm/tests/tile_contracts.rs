//! ROCm instantiation of the shared tiling conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_tile_contract;
use hephaestus_rocm::{RocmDevice, RocmTileOps};

#[test]
fn rocm_satisfies_the_tile_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm tile conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm tile conformance requires a physical device: {error}"),
    };
    assert_tile_contract::<_, _, i32>(&device, &RocmTileOps);
    assert_tile_contract::<_, _, f32>(&device, &RocmTileOps);
}
