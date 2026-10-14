//! CUDA instantiation of the shared scatter-add conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{assert_scatter_contract, assert_scatter_order_contract};
use hephaestus_cuda::{CudaDevice, CudaScatterOps};

#[test]
fn cuda_satisfies_the_scatter_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA scatter conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA scatter conformance requires a physical device: {error}"),
    };
    assert_scatter_contract::<_, _, i32>(&device, &CudaScatterOps);
    assert_scatter_contract::<_, _, f32>(&device, &CudaScatterOps);
    assert_scatter_order_contract(&device, &CudaScatterOps);
}
