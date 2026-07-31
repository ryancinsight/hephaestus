//! WGPU instantiation of the shared typed-elementwise conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value. The
//! clauses cover the scalar-aware comparison dispatch paths, four of the six
//! shared entry points no backend exercised before `ATLAS-ARCH-001`.

use hephaestus_conformance::assert_typed_elementwise_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuElementwiseOps};

#[test]
fn wgpu_satisfies_the_typed_elementwise_contract() {
    let device = match WgpuDevice::try_default("hephaestus-typed-elementwise-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU typed-elementwise conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_typed_elementwise_contract(&device, &WgpuElementwiseOps);
}
