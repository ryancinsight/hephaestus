//! Metal instantiation of the shared staggered-grid conformance clauses.
//!
//! `MetalStaggered3DOps` implements `Staggered3DOps` and, until this file
//! existed, no test in the crate mentioned it: the implementation and its
//! contract had drifted apart with nothing measuring the distance. The suite is
//! opt-in by construction, so an unwired clause is indistinguishable from a
//! satisfied one.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_staggered_3d_contract;
use hephaestus_metal::{MetalDevice, MetalStaggered3DOps};

#[test]
fn metal_satisfies_the_staggered_contract() {
    let device =
        MetalDevice::try_default().expect("Metal staggered conformance requires a physical device");
    assert_staggered_3d_contract(&device, &MetalStaggered3DOps);
}
