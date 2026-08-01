//! Metal instantiation of the shared transfer conformance clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_transfer_contract;
use hephaestus_metal::MetalDevice;

#[test]
fn metal_satisfies_the_transfer_contract() {
    let device =
        MetalDevice::try_default().expect("Metal transfer conformance requires a physical device");
    assert_transfer_contract(&device);
}
