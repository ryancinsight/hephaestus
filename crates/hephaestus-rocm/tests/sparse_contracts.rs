//! ROCm instantiation of the shared sparse-operator conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::SparseOperatorOps`]; this file only supplies the device
//! and the backend's seam value.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::{assert_batch_submit_contract, assert_sparse_operator_contract};
use hephaestus_rocm::{RocmDevice, RocmSparseOps};

#[test]
fn rocm_satisfies_the_sparse_operator_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm sparse conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm sparse conformance requires a physical device: {error}"),
    };
    assert_sparse_operator_contract(&device, &RocmSparseOps);
    assert_batch_submit_contract(&device, &RocmSparseOps);
}
