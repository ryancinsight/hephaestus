//! WGPU instantiation of the shared CTC clauses.

use hephaestus_conformance::assert_ctc_contract;
use hephaestus_core::DevicePreference;
use hephaestus_wgpu::{WgpuCtcOps, WgpuDevice};

fn ctc_device() -> Option<WgpuDevice> {
    // The posterior binds seven storage buffers plus the uniform buffer,
    // above the downlevel four the shared harness device offers, so this
    // case acquires its own raised device.
    let mut limits = WgpuDevice::downlevel_device_limits();
    limits.max_storage_buffers_per_shader_stage = Some(7);
    limits.max_buffers_and_acceleration_structures_per_shader_stage = Some(8);
    let required = std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some();
    match WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
        "hephaestus-ctc-contracts",
        DevicePreference::HighPerformance,
        &[],
        limits,
    ) {
        Ok(device) => {
            if device.limits().max_storage_buffers_per_shader_stage < 7 {
                if required {
                    panic!("CTC contract requires 7 storage buffers per shader stage");
                }
                eprintln!("skipping CTC contract: adapter offers fewer than 7 storage buffers");
                return None;
            }
            Some(device)
        }
        Err(error) => {
            if required {
                panic!("CTC contract requires a raised-limits adapter: {error}");
            }
            eprintln!("skipping CTC contract: {error}");
            None
        }
    }
}

pub(super) fn wgpu_satisfies_the_ctc_contract() {
    let Some(device) = ctc_device() else {
        return;
    };
    assert_ctc_contract(&device, &WgpuCtcOps);
}
