//! Metal instantiation of the shared fixed-scheme 3-D sweep clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_fixed_fd_3d_contract;
use hephaestus_metal::{MetalDevice, MetalFixedFd3DOps};

#[test]
fn metal_satisfies_the_fixed_fd_contract() {
    let device =
        MetalDevice::try_default().expect("Metal fixed-fd conformance requires a physical device");
    assert_fixed_fd_3d_contract(&device, &MetalFixedFd3DOps);
}
