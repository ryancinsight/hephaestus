//! CUDA instantiation of the shared gather conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_gather_contract;
use hephaestus_cuda::{CudaDevice, CudaGatherOps};

#[test]
fn cuda_satisfies_the_gather_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA gather conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA gather conformance requires a physical device: {error}"),
    };
    assert_gather_contract::<_, _, i32>(&device, &CudaGatherOps);
    assert_gather_contract::<_, _, f32>(&device, &CudaGatherOps);
}
