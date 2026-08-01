//! WGPU instantiation of the shared dense-product conformance clauses.

use hephaestus_conformance::assert_dense_product_contract;
use hephaestus_wgpu::{WgpuDenseProductOps, WgpuDevice};

#[test]
fn wgpu_satisfies_the_dense_product_contract() {
    let device = match WgpuDevice::try_default("hephaestus-dense-product-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU dense-product conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_dense_product_contract(&device, &WgpuDenseProductOps);
}
