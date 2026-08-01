//! WGPU instantiation of the shared random-initialization clauses.

#![cfg(any(feature = "decomposition", feature = "sparse"))]

use hephaestus_conformance::assert_random_init_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuRandomOps};

#[test]
fn wgpu_satisfies_the_random_init_contract() {
    let device = match WgpuDevice::try_default("hephaestus-random-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU random conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_random_init_contract(&device, &WgpuRandomOps);
}
