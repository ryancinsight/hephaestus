//! CUDA instantiation of the shared fixed-scheme 3-D sweep clauses.
//!
//! The shared conformance clause runs the dispatch lane-by-lane against
//! `leto_ops::FiniteDifference3D` — the CPU implementation this kernel
//! mirrors — over every scheme, every axis, and each scheme's minimum axis
//! extent, so every boundary fall-back branch is exercised on the device.

use hephaestus_conformance::assert_fixed_fd_3d_contract;
use hephaestus_cuda::{CudaDevice, CudaFixedFd3DOps};

fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_some() {
                panic!("CUDA device required for {test}: {error}");
            }
            eprintln!("skipping CUDA fixed-fd contract {test}: {error}");
            None
        }
    }
}

#[test]
fn cuda_satisfies_the_fixed_fd_contract() {
    let Some(device) = device("cuda_satisfies_the_fixed_fd_contract") else {
        return;
    };
    assert_fixed_fd_3d_contract(&device, &CudaFixedFd3DOps);
}
