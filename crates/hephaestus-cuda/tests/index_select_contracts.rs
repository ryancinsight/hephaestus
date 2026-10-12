//! CUDA instantiation of the shared index-selection conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_index_select_contract;
use hephaestus_cuda::{CudaDevice, CudaIndexSelectOps};

#[test]
fn cuda_satisfies_the_index_select_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA index_select conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA index_select conformance requires a physical device: {error}"),
    };
    assert_index_select_contract::<_, _, i32>(&device, &CudaIndexSelectOps);
    assert_index_select_contract::<_, _, f32>(&device, &CudaIndexSelectOps);
}
