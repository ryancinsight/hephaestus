//! CUDA instantiation of the shared stencil conformance clauses.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_stencil_contract;
use hephaestus_cuda::{CudaDevice, CudaStencilOps};

#[test]
fn cuda_satisfies_the_stencil_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA stencil conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA stencil conformance requires a physical device: {error}"),
    };
    assert_stencil_contract(&device, &CudaStencilOps);
}
