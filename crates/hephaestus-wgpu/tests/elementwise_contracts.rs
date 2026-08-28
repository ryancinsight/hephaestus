//! WGPU instantiation of the shared untyped elementwise clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::ElementwiseOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::assert_elementwise_contract;
use hephaestus_wgpu::WgpuElementwiseOps;

pub(super) fn wgpu_satisfies_the_elementwise_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_elementwise_contract(&device, &WgpuElementwiseOps);
}
