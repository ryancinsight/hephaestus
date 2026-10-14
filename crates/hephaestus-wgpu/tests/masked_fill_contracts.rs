//! WGPU instantiation of the shared masked-fill conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::MaskedFillOps`]; this file only supplies the device
//! and the backend's seam value.

use hephaestus_conformance::{assert_masked_fill_contract, assert_masked_fill_special_mask_values};
use hephaestus_wgpu::WgpuMaskedFillOps;

fn device_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
    super::device_or_skip()
}

pub(super) fn wgpu_satisfies_the_masked_fill_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_masked_fill_contract::<_, _, i32>(&device, &WgpuMaskedFillOps);
    assert_masked_fill_contract::<_, _, f32>(&device, &WgpuMaskedFillOps);
    assert_masked_fill_special_mask_values(&device, &WgpuMaskedFillOps);
}
