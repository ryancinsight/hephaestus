//! ROCm instantiation of the shared retained-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value per scalar.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_retained_reduction_contract;
use hephaestus_rocm::{RocmDevice, RocmVectorOps};

fn acquire_device(case: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!(
                "skip ROCm retained-reduction conformance ({case}): device unavailable ({error})"
            );
            None
        }
        Err(error) => {
            panic!("ROCm retained-reduction conformance requires a physical device: {error}")
        }
    }
}

#[test]
fn rocm_satisfies_the_retained_reduction_contract() {
    let Some(device) = acquire_device("f32") else {
        return;
    };
    let ops = RocmVectorOps::<f32>::new(&device).expect("vector kernels register");
    assert_retained_reduction_contract(&device, &ops);
}

#[test]
fn rocm_satisfies_the_retained_reduction_contract_f64() {
    let Some(device) = acquire_device("f64") else {
        return;
    };
    let ops = RocmVectorOps::<f64>::new(&device).expect("vector kernels register");
    assert_retained_reduction_contract(&device, &ops);
}
