//! ROCm instantiation of the shared dense-vector conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_dense_vector_contract;
use hephaestus_rocm::{RocmDevice, RocmVectorOps};

#[test]
fn rocm_satisfies_the_dense_vector_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm dense-vector conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm dense-vector conformance requires a physical device: {error}"),
    };
    let ops = RocmVectorOps::new(&device).expect("vector kernels register");
    assert_dense_vector_contract(&device, &ops);
}
