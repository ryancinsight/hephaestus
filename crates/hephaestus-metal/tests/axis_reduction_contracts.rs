//! Metal instantiation of the shared axis-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::AxisReductionOps`];
//! this file only supplies the device and the backend's seam value. The clauses
//! cover `prod_axis_into` and `prepare_reduce_axis_into`, two of the six shared
//! entry points no backend exercised before `ATLAS-ARCH-001`.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_axis_reduction_contract;
use hephaestus_metal::{MetalAxisReductionOps, MetalDevice};

#[test]
fn metal_satisfies_the_axis_reduction_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal axis-reduction conformance requires a physical device");
    assert_axis_reduction_contract(&device, &MetalAxisReductionOps::default());
}
