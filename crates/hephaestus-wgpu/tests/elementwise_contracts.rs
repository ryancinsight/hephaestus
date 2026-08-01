//! WGPU instantiation of the shared untyped elementwise clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_elementwise_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuElementwiseOps};

#[test]
fn wgpu_satisfies_the_elementwise_contract() {
    let device = match WgpuDevice::try_default("hephaestus-elementwise-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU elementwise conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_elementwise_contract(&device, &WgpuElementwiseOps);
}
