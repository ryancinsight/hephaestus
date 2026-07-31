//! Metal instantiation of the shared typed-elementwise conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value. The
//! clauses cover the scalar-aware comparison dispatch paths, four of the six
//! shared entry points no backend exercised before `ATLAS-ARCH-001`.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_typed_elementwise_contract;
use hephaestus_metal::{MetalDevice, MetalElementwiseOps};

#[test]
fn metal_satisfies_the_typed_elementwise_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal typed-elementwise conformance requires a physical device");
    assert_typed_elementwise_contract(&device, &MetalElementwiseOps::default());
}
