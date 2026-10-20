//! CUDA instantiation of the shared retained-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value per scalar.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_retained_reduction_contract;
use hephaestus_cuda::{CudaDevice, CudaVectorOps};

fn acquire_device(case: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!(
                "skip CUDA retained-reduction conformance ({case}): device unavailable ({error})"
            );
            None
        }
        Err(error) => {
            panic!("CUDA retained-reduction conformance requires a physical device: {error}")
        }
    }
}

#[test]
fn cuda_satisfies_the_retained_reduction_contract() {
    let Some(device) = acquire_device("f32") else {
        return;
    };
    let ops = CudaVectorOps::<f32>::new(&device).expect("vector kernels register");
    assert_retained_reduction_contract(&device, &ops);
}

#[test]
fn cuda_satisfies_the_retained_reduction_contract_f64() {
    let Some(device) = acquire_device("f64") else {
        return;
    };
    let ops = CudaVectorOps::<f64>::new(&device).expect("vector kernels register");
    assert_retained_reduction_contract(&device, &ops);
}
