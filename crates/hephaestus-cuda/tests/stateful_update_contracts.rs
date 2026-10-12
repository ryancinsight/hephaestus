//! CUDA instantiation of the shared stateful-update contract.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_stateful_update_contract;
use hephaestus_cuda::{CudaDevice, CudaStatefulUpdateOps};

#[test]
fn cuda_satisfies_the_stateful_update_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA stateful-update conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA stateful-update conformance requires a device: {error}"),
    };
    assert_stateful_update_contract(&device, &CudaStatefulUpdateOps);
}
