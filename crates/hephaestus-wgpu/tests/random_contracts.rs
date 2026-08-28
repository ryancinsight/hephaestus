//! WGPU instantiation of the shared random-initialization clauses.

#![cfg(any(feature = "decomposition", feature = "sparse"))]

use hephaestus_conformance::assert_random_init_contract;
use hephaestus_wgpu::WgpuRandomOps;

pub(super) fn wgpu_satisfies_the_random_init_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_random_init_contract(&device, &WgpuRandomOps);
}
