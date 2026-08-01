//! Metal instantiation of the shared sparse-operator conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::SparseOperatorOps`]; this file only supplies the device
//! and the backend's seam value. CUDA and ROCm instantiate when they gain
//! native CSR SpMV implementations (ATLAS-ARCH-001c).

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_sparse_operator_contract;
use hephaestus_metal::{MetalDevice, MetalSparseOps};

#[test]
fn metal_satisfies_the_sparse_operator_contract() {
    let device =
        MetalDevice::try_default().expect("Metal sparse conformance requires a physical device");
    assert_sparse_operator_contract(&device, &MetalSparseOps::default());
}
