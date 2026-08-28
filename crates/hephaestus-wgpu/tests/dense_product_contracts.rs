//! WGPU instantiation of the shared dense-product conformance clauses.

use hephaestus_conformance::assert_dense_product_contract;
use hephaestus_wgpu::WgpuDenseProductOps;

pub(super) fn wgpu_satisfies_the_dense_product_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_dense_product_contract(&device, &WgpuDenseProductOps);
}
