//! WGPU instantiation of the shared roll conformance clause.

use hephaestus_conformance::assert_roll_contract;
use hephaestus_wgpu::WgpuRollOps;

pub(super) fn wgpu_satisfies_the_roll_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_roll_contract(&device, &WgpuRollOps);
}
