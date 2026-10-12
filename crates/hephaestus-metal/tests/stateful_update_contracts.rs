//! Metal instantiation of the shared stateful-update contract.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_stateful_update_contract;
use hephaestus_metal::{MetalDevice, MetalStatefulUpdateOps};

#[test]
fn metal_satisfies_the_stateful_update_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal stateful-update conformance requires a physical device");
    assert_stateful_update_contract(&device, &MetalStatefulUpdateOps);
}
