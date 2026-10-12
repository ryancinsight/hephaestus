//! WGSL f64 cross-entropy backward contract.
//!
//! Forward stays out of this binary: the f64 forward shaders need `log`,
//! which aborts shader compilation on some Vulkan drivers, so only the
//! backward path (pure IEEE-double arithmetic) executes here. CUDA proves
//! the f64 forward path on hardware.

use hephaestus_conformance::assert_cross_entropy_backward_contract_f64;
use hephaestus_core::{DeviceFeature, DevicePreference};
use hephaestus_wgpu::{WgpuCrossEntropyOps, WgpuDevice};

fn f64_device() -> Option<WgpuDevice> {
    // The f64 arithmetic preflight binds five storage buffers plus the
    // uniform buffer, above the downlevel four.
    let mut limits = WgpuDevice::downlevel_device_limits();
    limits.max_storage_buffers_per_shader_stage = Some(5);
    limits.max_buffers_and_acceleration_structures_per_shader_stage = Some(6);
    match WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
        "hephaestus-ce-f64-contracts",
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF64],
        limits,
    ) {
        Ok(device) => {
            if !device.supports_device_feature(DeviceFeature::ShaderF64) {
                eprintln!("contract skip: adapter lacks ShaderF64");
                return None;
            }
            if device.limits().max_storage_buffers_per_shader_stage < 5 {
                eprintln!("contract skip: adapter offers fewer than 5 storage buffers");
                return None;
            }
            Some(device)
        }
        Err(error) => {
            eprintln!("contract skip: {error}");
            None
        }
    }
}

#[test]
fn f64_backward_matches_the_leto_oracle() {
    let Some(device) = f64_device() else { return };
    assert_cross_entropy_backward_contract_f64(&device, &WgpuCrossEntropyOps);
}
