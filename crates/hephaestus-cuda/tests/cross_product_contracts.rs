//! CUDA instantiation of the shared cross-product conformance clause.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_cross_product_contract;
use hephaestus_cuda::{CudaCrossProductOps, CudaDevice};

#[test]
fn cuda_satisfies_the_cross_product_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA cross-product conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA cross-product conformance requires a physical device: {error}"),
    };
    assert_cross_product_contract(&device, &CudaCrossProductOps);
}
