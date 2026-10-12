//! Metal instantiation of the shared CTC clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_ctc_contract;
use hephaestus_core::{ComputeDeviceAcquisition, DevicePreference};
use hephaestus_metal::{MetalCtcOps, MetalDevice};
use hephaestus_wgpu::WgpuDevice;

#[test]
fn metal_satisfies_the_ctc_contract() {
    // The posterior binds seven storage buffers plus the uniform buffer,
    // above the downlevel four `MetalDevice::try_default` requests, so this
    // case acquires its own raised device.
    let mut limits = WgpuDevice::downlevel_device_limits();
    limits.max_storage_buffers_per_shader_stage = Some(7);
    limits.max_buffers_and_acceleration_structures_per_shader_stage = Some(8);
    let device = MetalDevice::try_acquire_device(
        "hephaestus-metal-ctc-contracts",
        DevicePreference::HighPerformance,
        &[],
        limits,
    )
    .expect("Metal CTC conformance requires a physical device with raised storage limits");
    assert_ctc_contract(&device, &MetalCtcOps);
}
