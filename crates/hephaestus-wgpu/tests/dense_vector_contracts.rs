//! WGPU instantiation of the shared dense-vector conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_dense_vector_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuVectorOps};

#[test]
fn wgpu_satisfies_the_dense_vector_contract() {
    let device = match WgpuDevice::try_default("hephaestus-dense-vector-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU dense-vector conformance: adapter unavailable ({error})");
            return;
        }
    };
    let ops = WgpuVectorOps::new(&device).expect("vector kernels compile");
    assert_dense_vector_contract(&device, &ops);
}
