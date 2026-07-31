//! CUDA instantiation of the shared typed-elementwise conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value. The
//! clauses cover the scalar-aware comparison dispatch paths, four of the six
//! shared entry points no backend exercised before `ATLAS-ARCH-001`.

#![cfg(feature = "cuda")]

use hephaestus_conformance::assert_typed_elementwise_contract;
use hephaestus_cuda::{CudaDevice, CudaElementwiseOps};

#[test]
fn cuda_satisfies_the_typed_elementwise_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA typed-elementwise conformance: device unavailable ({error})");
            return;
        }
        Err(error) => {
            panic!("CUDA typed-elementwise conformance requires a physical device: {error}")
        }
    };
    assert_typed_elementwise_contract(&device, &CudaElementwiseOps);
}
