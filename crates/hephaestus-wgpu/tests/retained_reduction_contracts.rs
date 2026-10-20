//! WGPU instantiation of the shared retained-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value per scalar.

use hephaestus_conformance::assert_retained_reduction_contract;
use hephaestus_core::{DeviceFeature, DevicePreference};
use hephaestus_wgpu::{WgpuDevice, WgpuVectorOps};

pub(super) fn wgpu_satisfies_the_retained_reduction_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::<f32>::new(&device).expect("vector kernels compile");
    assert_retained_reduction_contract(&device, &ops);
}

pub(super) fn wgpu_satisfies_the_retained_reduction_contract_f64() {
    let Some(device) = f64_device_or_skip() else {
        return;
    };
    let ops = WgpuVectorOps::<f64>::new(&device).expect("vector kernels compile");
    assert_retained_reduction_contract(&device, &ops);
}

/// Acquire a ShaderF64-capable device, skipping when the adapter lacks it.
fn f64_device_or_skip() -> Option<WgpuDevice> {
    let device = WgpuDevice::try_with_device_preference_and_optional_device_features(
        "hephaestus-vector-f64",
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF64],
    )
    .ok()?;
    if device.supports_device_feature(DeviceFeature::ShaderF64) {
        Some(device)
    } else {
        eprintln!("skipping WGPU f64 retained-reduction conformance: ShaderF64 unavailable");
        None
    }
}
