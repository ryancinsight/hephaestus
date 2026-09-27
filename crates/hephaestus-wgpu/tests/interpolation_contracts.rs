//! WGPU instantiation of the shared interpolation conformance clause.

use hephaestus_conformance::assert_interpolation_contract;
use hephaestus_wgpu::WgpuInterpolationOps;

pub(super) fn wgpu_satisfies_the_interpolation_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_interpolation_contract(&device, &WgpuInterpolationOps);
}
