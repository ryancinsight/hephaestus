//! WGPU instantiation of the shared dense-product conformance clauses.

#[cfg(any(feature = "decomposition", feature = "sparse"))]
use hephaestus_conformance::assert_dense_matrix_function_contract;
use hephaestus_conformance::{assert_dense_composition_contract, assert_dense_product_contract};
use hephaestus_wgpu::WgpuDenseProductOps;

pub(super) fn wgpu_satisfies_the_dense_product_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_dense_product_contract(&device, &WgpuDenseProductOps);
    assert_dense_composition_contract(&device, &WgpuDenseProductOps);
    #[cfg(any(feature = "decomposition", feature = "sparse"))]
    assert_dense_matrix_function_contract(&device, &WgpuDenseProductOps);
}
