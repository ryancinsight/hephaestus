//! ROCm instantiation of the shared stencil conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hephaestus_conformance::assert_stencil_contract;
use hephaestus_rocm::{RocmDevice, RocmStencilOps};

#[test]
fn rocm_satisfies_the_stencil_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm stencil conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm stencil conformance requires a physical device: {error}"),
    };
    assert_stencil_contract(&device, &RocmStencilOps);
}
