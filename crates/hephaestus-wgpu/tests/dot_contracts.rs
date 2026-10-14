//! WGPU instantiation of the shared inner-product conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DotOps`];
//! this file only supplies the device and the backend's seam value.

use hephaestus_conformance::{assert_dot_contract, assert_dot_special_float_values};
use hephaestus_core::{DeviceFeature, DevicePreference};
use hephaestus_wgpu::{WgpuDevice, WgpuDotOps};

pub(super) fn wgpu_satisfies_the_dot_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_dot_contract::<_, _, i32>(&device, &WgpuDotOps);
    assert_dot_contract::<_, _, f32>(&device, &WgpuDotOps);
    assert_dot_special_float_values(&device, &WgpuDotOps);
}

pub(super) fn wgpu_satisfies_the_dot_contract_f64() {
    let Some(device) = f64_device_or_skip() else {
        return;
    };
    assert_dot_contract::<_, _, f64>(&device, &WgpuDotOps);
}

/// Acquire a ShaderF64-capable device, skipping when the adapter lacks it.
fn f64_device_or_skip() -> Option<WgpuDevice> {
    let device = WgpuDevice::try_with_device_preference_and_optional_device_features(
        "hephaestus-dot-f64",
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF64],
    )
    .ok()?;
    if device.supports_device_feature(DeviceFeature::ShaderF64) {
        Some(device)
    } else {
        eprintln!("skipping WGPU f64 dot conformance: ShaderF64 unavailable");
        None
    }
}
