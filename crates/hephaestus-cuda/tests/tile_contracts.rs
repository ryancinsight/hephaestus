//! CUDA instantiation of the shared tiling conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_tile_contract;
use hephaestus_cuda::{CudaDevice, CudaTileOps};

#[test]
fn cuda_satisfies_the_tile_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA tile conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA tile conformance requires a physical device: {error}"),
    };
    assert_tile_contract::<_, _, i32>(&device, &CudaTileOps);
    assert_tile_contract::<_, _, f32>(&device, &CudaTileOps);
}
