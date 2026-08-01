//! WGPU instantiation of the shared decomposition conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::DecompositionOps`]; this file only supplies the device
//! and the backend seam value.

use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_wgpu::{WgpuDecompositionOps, WgpuDevice};

#[test]
fn wgpu_satisfies_the_decomposition_contract() {
    let device = match WgpuDevice::try_default("hephaestus-decomposition-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU decomposition conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_decomposition_contract(&device, &WgpuDecompositionOps);
}
