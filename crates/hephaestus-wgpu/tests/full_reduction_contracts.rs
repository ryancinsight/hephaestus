//! WGPU instantiation of the shared full-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::FullReductionOps`]; this file only supplies the device
//! and the backend's seam value.

use hephaestus_conformance::assert_full_reduction_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuFullReductionOps};

#[test]
fn wgpu_satisfies_the_full_reduction_contract() {
    let device = match WgpuDevice::try_default("hephaestus-full-reduction-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU full-reduction conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_full_reduction_contract(&device, &WgpuFullReductionOps);
}
