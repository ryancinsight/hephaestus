//! WGPU instantiation of the shared sparse-operator conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::SparseOperatorOps`]; this file only supplies the device
//! and the backend's seam value. CUDA and ROCm instantiate when they gain
//! native CSR SpMV implementations (ATLAS-ARCH-001c).

use hephaestus_conformance::{assert_batch_submit_contract, assert_sparse_operator_contract};
use hephaestus_wgpu::WgpuSparseOps;

pub(super) fn wgpu_satisfies_the_sparse_operator_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_sparse_operator_contract(&device, &WgpuSparseOps);
    assert_batch_submit_contract(&device, &WgpuSparseOps);
}
