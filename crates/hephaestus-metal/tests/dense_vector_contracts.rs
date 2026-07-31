//! Metal instantiation of the shared dense-vector conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_dense_vector_contract;
use hephaestus_metal::{MetalDevice, MetalVectorOps};

#[test]
fn metal_satisfies_the_dense_vector_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal dense-vector conformance requires a physical device");
    let ops = MetalVectorOps::new(&device).expect("vector kernels compile");
    assert_dense_vector_contract(&device, &ops);
}
