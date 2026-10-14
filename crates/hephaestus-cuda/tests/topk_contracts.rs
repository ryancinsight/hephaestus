//! CUDA instantiation of the shared top-k conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{assert_topk_contract, assert_topk_strided_output_contract};
use hephaestus_cuda::{CudaDevice, CudaTopKOps};

#[test]
fn cuda_satisfies_the_topk_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA top-k conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA top-k conformance requires a physical device: {error}"),
    };
    assert_topk_contract(&device, &CudaTopKOps);
    assert_topk_strided_output_contract(&device, &CudaTopKOps);
}
