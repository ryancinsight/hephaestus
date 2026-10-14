//! CUDA instantiation of the shared pad conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_pad_contract;
use hephaestus_cuda::{CudaDevice, CudaPadOps};

#[test]
fn cuda_satisfies_the_pad_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA pad conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA pad conformance requires a physical device: {error}"),
    };
    assert_pad_contract::<_, _, i32>(&device, &CudaPadOps);
    assert_pad_contract::<_, _, f32>(&device, &CudaPadOps);
}
