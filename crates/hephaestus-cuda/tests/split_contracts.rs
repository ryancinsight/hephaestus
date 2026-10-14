//! CUDA instantiation of the shared splitting conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_split_contract;
use hephaestus_cuda::{CudaDevice, CudaSplitOps};

#[test]
fn cuda_satisfies_the_split_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA split conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA split conformance requires a physical device: {error}"),
    };
    assert_split_contract::<_, _, i32>(&device, &CudaSplitOps);
    assert_split_contract::<_, _, f32>(&device, &CudaSplitOps);
}
