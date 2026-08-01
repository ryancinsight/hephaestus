//! WGPU instantiation of the shared stateful-update contract.

use hephaestus_conformance::assert_stateful_update_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuStatefulUpdateOps};

#[test]
fn wgpu_satisfies_the_stateful_update_contract() {
    let device = match WgpuDevice::try_default("hephaestus-stateful-update-test") {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip WGPU stateful-update conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("WGPU stateful-update conformance requires a device: {error}"),
    };
    assert_stateful_update_contract(&device, &WgpuStatefulUpdateOps);
}
