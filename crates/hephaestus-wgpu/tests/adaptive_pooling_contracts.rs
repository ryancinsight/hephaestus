//! WGPU instantiation of the shared adaptive-pooling conformance clause.

use hephaestus_conformance::assert_adaptive_pooling_contract;
use hephaestus_wgpu::WgpuAdaptivePoolingOps;

pub(super) fn wgpu_satisfies_the_adaptive_pooling_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_adaptive_pooling_contract(&device, &WgpuAdaptivePoolingOps);
}
