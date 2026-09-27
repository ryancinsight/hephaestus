//! WGPU instantiation of the shared cross-product conformance clause.

use hephaestus_conformance::assert_cross_product_contract;
use hephaestus_wgpu::WgpuCrossProductOps;

pub(super) fn wgpu_satisfies_the_cross_product_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_cross_product_contract(&device, &WgpuCrossProductOps);
}
