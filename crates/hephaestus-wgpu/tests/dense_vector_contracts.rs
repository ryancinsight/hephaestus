//! WGPU instantiation of the shared dense-vector conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_dense_vector_contract;
use hephaestus_wgpu::WgpuVectorOps;

pub(super) fn wgpu_satisfies_the_dense_vector_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::new(&device).expect("vector kernels compile");
    assert_dense_vector_contract(&device, &ops);
}
