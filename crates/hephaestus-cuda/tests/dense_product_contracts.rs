//! CUDA instantiation of the shared dense-product conformance clauses.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_dense_product_contract;
use hephaestus_cuda::{CudaDenseProductOps, CudaDevice};

#[test]
fn cuda_satisfies_the_dense_product_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA dense-product conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA dense-product conformance requires a physical device: {error}"),
    };
    assert_dense_product_contract(&device, &CudaDenseProductOps);
}
