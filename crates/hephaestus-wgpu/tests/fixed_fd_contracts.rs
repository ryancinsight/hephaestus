//! WGPU instantiation of the shared fixed-scheme 3-D sweep clauses.

use hephaestus_conformance::assert_fixed_fd_3d_contract;
use hephaestus_wgpu::WgpuFixedFd3DOps;

pub(super) fn wgpu_satisfies_the_fixed_fd_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_fixed_fd_3d_contract(&device, &WgpuFixedFd3DOps);
}
