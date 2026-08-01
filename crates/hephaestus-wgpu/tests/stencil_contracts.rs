//! WGPU instantiation of the shared stencil conformance clauses.

use hephaestus_conformance::assert_stencil_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuStencilOps};

#[test]
fn wgpu_satisfies_the_stencil_contract() {
    let device = match WgpuDevice::try_default("hephaestus-stencil-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU stencil conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_stencil_contract(&device, &WgpuStencilOps);
}
