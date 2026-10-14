//! CUDA instantiation of the shared interpolation conformance clause.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_interpolation_contract;
use hephaestus_cuda::{CudaDevice, CudaInterpolationOps};

#[test]
fn cuda_satisfies_the_interpolation_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA interpolation conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA interpolation conformance requires a physical device: {error}"),
    };
    assert_interpolation_contract(&device, &CudaInterpolationOps);
}
