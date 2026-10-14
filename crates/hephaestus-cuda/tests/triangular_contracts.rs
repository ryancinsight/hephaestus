//! CUDA instantiation of the shared triangular conformance clause.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_triangular_contract;
use hephaestus_cuda::{CudaDevice, CudaTriangularOps};

#[test]
fn cuda_satisfies_the_triangular_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA triangular conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA triangular conformance requires a physical device: {error}"),
    };
    assert_triangular_contract(&device, &CudaTriangularOps);
}
