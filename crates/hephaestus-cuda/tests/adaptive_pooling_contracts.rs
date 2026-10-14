//! CUDA instantiation of the shared adaptive-pooling conformance clause.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_adaptive_pooling_contract;
use hephaestus_cuda::{CudaAdaptivePoolingOps, CudaDevice};

#[test]
fn cuda_satisfies_the_adaptive_pooling_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA adaptive-pooling conformance: device unavailable ({error})");
            return;
        }
        Err(error) => {
            panic!("CUDA adaptive-pooling conformance requires a physical device: {error}")
        }
    };
    assert_adaptive_pooling_contract(&device, &CudaAdaptivePoolingOps);
}
