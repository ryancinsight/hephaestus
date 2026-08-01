//! Metal instantiation of the shared untyped elementwise clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_elementwise_contract;
use hephaestus_metal::{MetalDevice, MetalElementwiseOps};

#[test]
fn metal_satisfies_the_elementwise_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal elementwise conformance requires a physical device");
    assert_elementwise_contract(&device, &MetalElementwiseOps::default());
}
