//! WGPU instantiation of the shared nonzero-coordinate conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::NonzeroOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::{
    assert_nonzero_contract, assert_nonzero_rejects_aliased_output,
    assert_nonzero_special_float_values,
};
use hephaestus_wgpu::WgpuNonzeroOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_nonzero_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_nonzero_contract::<_, _, i32>(&device, &WgpuNonzeroOps);
    assert_nonzero_contract::<_, _, f32>(&device, &WgpuNonzeroOps);
    assert_nonzero_special_float_values(&device, &WgpuNonzeroOps);
    assert_nonzero_rejects_aliased_output(&device, &WgpuNonzeroOps);
}
