//! Metal instantiation of the shared CTC clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_ctc_contract;
use hephaestus_metal::{MetalCtcOps, MetalDevice};

#[test]
fn metal_satisfies_the_ctc_contract() {
    let device =
        MetalDevice::try_default().expect("Metal CTC conformance requires a physical device");
    assert_ctc_contract(&device, &MetalCtcOps);
}
