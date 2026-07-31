//! Metal instantiation of the shared full-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::FullReductionOps`]; this file only supplies the device
//! and the backend's seam value.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_full_reduction_contract;
use hephaestus_metal::{MetalDevice, MetalFullReductionOps};

#[test]
fn metal_satisfies_the_full_reduction_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal full-reduction conformance requires a physical device");
    assert_full_reduction_contract(&device, &MetalFullReductionOps::default());
}
