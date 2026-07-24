//! Value-semantic contracts for the ROCm device substrate.
//!
//! The default build verifies the typed unavailable path. With `--features
//! rocm`, the same tests execute against HIP when an AMD device is present.
//! Hardware CI sets `HEPHAESTUS_ROCM_REQUIRE_DEVICE=1` so an unavailable device
//! fails that lane instead of being reported as device evidence.

use hephaestus_core::{
    ComputeDevice, ComputeDeviceCapabilities, DeviceBuffer, DeviceFeature, HephaestusError,
};
use hephaestus_rocm::{Result, RocmDevice};

fn device(test: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_some() {
                panic!("ROCm device required for {test}, but acquisition failed: {error}");
            }
            eprintln!("skip {test}: ROCm device unavailable ({error})");
            None
        }
    }
}

fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
    match result {
        Err(HephaestusError::LengthMismatch {
            host_len: actual_host_len,
            device_len: actual_device_len,
        }) => {
            assert_eq!(actual_host_len, host_len);
            assert_eq!(actual_device_len, device_len);
        }
        Err(error) => panic!("expected length mismatch, got {error:?}"),
        Ok(_) => panic!("expected length mismatch {host_len}->{device_len}, got success"),
    }
}

#[test]
fn backend_name_is_stable() {
    #[cfg(not(feature = "rocm"))]
    assert_eq!(
        RocmDevice::try_default().unwrap_err().to_string(),
        "no compatible accelerator adapter available: ROCm backend unavailable: enable the `rocm` feature on Linux with ROCm installed"
    );

    #[cfg(feature = "rocm")]
    if let Some(device) = device("backend_name_is_stable") {
        assert_eq!(device.backend_name(), "rocm");
    }
}

#[test]
fn device_capabilities_and_topology_are_driver_backed() {
    let Some(device) = device("device_capabilities_and_topology_are_driver_backed") else {
        return;
    };

    let limits = device.device_limits();
    assert!(limits.max_buffer_size > 0);
    assert!(limits.max_compute_workgroup_size_x > 0);
    assert!(limits.max_compute_workgroup_size_y > 0);
    assert!(limits.max_compute_workgroup_size_z > 0);
    assert!(limits.max_compute_invocations_per_workgroup > 0);
    assert!(limits.max_compute_workgroup_storage_size > 0);
    assert_eq!(limits.max_storage_buffers_per_shader_stage, None);
    assert_eq!(
        limits.max_buffers_and_acceleration_structures_per_shader_stage,
        None
    );
    assert_eq!(limits.max_immediate_size, 0);

    assert!(!device.supports_device_feature(DeviceFeature::TimestampQuery));
    assert!(!device.supports_device_feature(DeviceFeature::ShaderF64));
    assert!(!device.supports_device_feature(DeviceFeature::ShaderF16));
    assert!(!device.supports_device_feature(DeviceFeature::ImmediateData));

    let topology = device
        .topology()
        .expect("an acquired ROCm device must have a topology snapshot");
    assert!(topology.compute_units() > 0);
    assert!(topology.warp_width() > 0);
    assert!(topology.max_threads_per_unit() > 0);
    assert!(topology.registers_per_unit() > 0);
    assert!(topology.shared_mem_per_unit_bytes() > 0);
    assert!(topology.memory_bytes() > 0);
    assert_eq!(topology.memory_tier(), themis::MemoryTier::Device);
    assert!(topology.max_resident_warps() > 0);
}

#[test]
fn upload_download_roundtrip_preserves_values() {
    let Some(device) = device("upload_download_roundtrip_preserves_values") else {
        return;
    };

    let host = [1.0_f32, 2.0, -3.5, 4.25, 0.0, 1024.5];
    let buffer = device.upload(&host).expect("HIP upload");
    assert_eq!(buffer.len(), host.len());
    assert_eq!(buffer.tier(), themis::MemoryTier::Device);

    let mut output = [0.0_f32; 6];
    device.download(&buffer, &mut output).expect("HIP download");
    device.synchronize().expect("HIP synchronization");
    assert_eq!(output, host);
}

#[test]
fn alloc_zeroed_produces_zero_values() {
    let Some(device) = device("alloc_zeroed_produces_zero_values") else {
        return;
    };

    let buffer = device.alloc_zeroed::<u32>(17).expect("HIP allocation");
    let mut output = [9_u32; 17];
    device.download(&buffer, &mut output).expect("HIP download");
    assert_eq!(output, [0_u32; 17]);
}

#[test]
fn placement_hints_match_hip_allocation_contract() {
    let Some(device) = device("placement_hints_match_hip_allocation_contract") else {
        return;
    };

    let host_visible = device
        .alloc_zeroed_with_hint::<u32>(
            4,
            themis::PlacementHint::Tier(themis::MemoryTier::HostPinned),
        )
        .expect("host-visible hints normalize to HIP device memory");
    assert_eq!(host_visible.tier(), themis::MemoryTier::Device);

    let registers = device.alloc_zeroed_with_hint::<u32>(
        4,
        themis::PlacementHint::Tier(themis::MemoryTier::Registers),
    );
    assert!(matches!(
        registers,
        Err(HephaestusError::AllocationFailed { message })
            if message == "ROCm primary buffers cannot be allocated from budget-only tier Registers"
    ));
}

#[test]
fn subrange_write_preserves_unwritten_values() {
    let Some(device) = device("subrange_write_preserves_unwritten_values") else {
        return;
    };

    let buffer = device.upload(&[1_u32, 2, 3, 4, 5]).expect("HIP upload");
    device
        .write_sub_buffer(&buffer, 2, &[9_u32, 8])
        .expect("HIP subrange write");
    let mut output = [0_u32; 5];
    device.download(&buffer, &mut output).expect("HIP download");
    assert_eq!(output, [1, 2, 9, 8, 5]);
}

#[test]
fn length_mismatches_are_rejected_before_transfer() {
    let Some(device) = device("length_mismatches_are_rejected_before_transfer") else {
        return;
    };

    let buffer = device.upload(&[1.0_f32, 2.0]).expect("HIP upload");
    let mut output = [0.0_f32; 3];
    assert_length_mismatch(device.download(&buffer, &mut output), 3, 2);
    assert_length_mismatch(device.write_buffer(&buffer, &[1.0_f32]), 1, 2);
}

#[test]
fn empty_buffers_roundtrip_without_hip_allocation() {
    let Some(device) = device("empty_buffers_roundtrip_without_hip_allocation") else {
        return;
    };

    let buffer = device.upload::<u16>(&[]).expect("empty HIP upload");
    assert!(buffer.is_empty());
    let mut output = [];
    device
        .download(&buffer, &mut output)
        .expect("empty HIP download");
}
