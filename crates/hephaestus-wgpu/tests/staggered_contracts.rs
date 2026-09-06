//! WGPU instantiation of the shared staggered conformance clauses.

use hephaestus_conformance::assert_staggered_3d_contract;
use hephaestus_wgpu::WgpuStaggered3DOps;

pub(super) fn wgpu_satisfies_the_staggered_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_staggered_3d_contract(&device, &WgpuStaggered3DOps);
}
