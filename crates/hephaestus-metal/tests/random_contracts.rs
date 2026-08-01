//! Metal instantiation of the shared random-initialization clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_random_init_contract;
use hephaestus_metal::{MetalDevice, MetalRandomOps};

#[test]
fn metal_satisfies_the_random_init_contract() {
    let device =
        MetalDevice::try_default().expect("Metal random conformance requires a physical device");
    assert_random_init_contract(&device, &MetalRandomOps);
}
