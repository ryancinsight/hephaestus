//! WGPU instantiation of the shared stencil conformance clauses.

use hephaestus_conformance::assert_stencil_contract;
use hephaestus_wgpu::WgpuStencilOps;

pub(super) fn wgpu_satisfies_the_stencil_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_stencil_contract(&device, &WgpuStencilOps);
}
