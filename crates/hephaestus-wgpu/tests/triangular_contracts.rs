//! WGPU instantiation of the shared triangular-masking conformance clause.

use hephaestus_conformance::assert_triangular_contract;
use hephaestus_wgpu::WgpuTriangularOps;

pub(super) fn wgpu_satisfies_the_triangular_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_triangular_contract(&device, &WgpuTriangularOps);
}
