//! ROCm instantiation of the shared axis-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::AxisReductionOps`];
//! this file only supplies the device and the backend's seam value. The clauses
//! cover `prod_axis_into` and `prepare_reduce_axis_into`, two of the six shared
//! entry points no backend exercised before `ATLAS-ARCH-001`.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_axis_reduction_contract;
use hephaestus_rocm::{RocmAxisReductionOps, RocmDevice};

#[test]
fn rocm_satisfies_the_axis_reduction_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm axis-reduction conformance: device unavailable ({error})");
            return;
        }
        Err(error) => {
            panic!("ROCm axis-reduction conformance requires a physical device: {error}")
        }
    };
    assert_axis_reduction_contract(&device, &RocmAxisReductionOps);
}
