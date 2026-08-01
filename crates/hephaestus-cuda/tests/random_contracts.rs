//! CUDA instantiation of the shared random-initialization clauses.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_random_init_contract;
use hephaestus_cuda::{CudaDevice, CudaRandomOps};

#[test]
fn cuda_satisfies_the_random_init_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA random conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA random conformance requires a physical device: {error}"),
    };
    assert_random_init_contract(&device, &CudaRandomOps);
}
