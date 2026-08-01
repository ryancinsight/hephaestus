//! Metal instantiation of the shared stencil conformance clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_stencil_contract;
use hephaestus_metal::{MetalDevice, MetalStencilOps};

#[test]
fn metal_satisfies_the_stencil_contract() {
    let device =
        MetalDevice::try_default().expect("Metal stencil conformance requires a physical device");
    assert_stencil_contract(&device, &MetalStencilOps);
}
