//! CUDA instantiation of the shared transfer conformance clauses.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_transfer_contract;
use hephaestus_cuda::CudaDevice;

#[test]
fn cuda_satisfies_the_transfer_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA transfer conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA transfer conformance requires a physical device: {error}"),
    };
    assert_transfer_contract(&device);
}
