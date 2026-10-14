//! CUDA instantiation of the shared circular-shift conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_roll_contract;
use hephaestus_cuda::{CudaDevice, CudaRollOps};

#[test]
fn cuda_satisfies_the_roll_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA roll conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA roll conformance requires a physical device: {error}"),
    };
    assert_roll_contract::<_, _, i32>(&device, &CudaRollOps);
    assert_roll_contract::<_, _, f32>(&device, &CudaRollOps);
}
