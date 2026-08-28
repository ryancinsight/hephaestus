//! WGPU instantiation of the shared runtime-parameter unary contract.

use hephaestus_conformance::assert_parameterized_unary_contract;
use hephaestus_wgpu::WgpuParameterizedUnaryOps;

pub(super) fn wgpu_satisfies_the_parameterized_unary_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_parameterized_unary_contract(&device, &WgpuParameterizedUnaryOps);
}
