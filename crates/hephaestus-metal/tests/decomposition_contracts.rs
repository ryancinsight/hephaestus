//! Metal instantiation of the shared decomposition conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::DecompositionOps`]; this file only supplies the device
//! and the backend seam value.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_metal::{MetalDecompositionOps, MetalDevice};

#[test]
fn metal_satisfies_the_decomposition_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal decomposition conformance requires a physical device");
    assert_decomposition_contract(&device, &MetalDecompositionOps);
}
