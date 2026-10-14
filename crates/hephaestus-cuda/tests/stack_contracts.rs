//! CUDA instantiation of the shared stacking conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_stack_contract;
use hephaestus_cuda::{CudaDevice, CudaStackOps};

#[test]
fn cuda_satisfies_the_stack_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA stack conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA stack conformance requires a physical device: {error}"),
    };
    assert_stack_contract::<_, _, i32>(&device, &CudaStackOps);
    assert_stack_contract::<_, _, f32>(&device, &CudaStackOps);
}
