use eunomia::{Pod, Zeroable};
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
use hephaestus_wgpu::WgpuDevice;

use super::device_or_skip;

fn upload_values<T: Pod + Zeroable + PartialEq + core::fmt::Debug>(
    device: &WgpuDevice,
    value: T,
) -> hephaestus_core::Result<()> {
    for len in [0, 1, 2, 3, 4, 5, 8] {
        let host: Vec<_> = (0..len)
            .map(|index| if index % 2 == 0 { value } else { T::zeroed() })
            .collect();
        let uploaded = device.upload(&host)?;
        assert_eq!(uploaded.len(), len);
        assert_eq!(device.download_owned(&uploaded)?, host);
        let zeroed = device.alloc_zeroed::<T>(len)?;
        assert_eq!(device.download_owned(&zeroed)?, vec![T::zeroed(); len]);
    }
    Ok(())
}

#[test]
fn storage_upload_preserves_empty_odd_and_aligned_payloads() {
    let Some(device) = device_or_skip() else {
        return;
    };
    upload_values(&device, 0xadu8).unwrap();
    upload_values(&device, 0xad19u16).unwrap();
    upload_values(&device, 0xad19_5832u32).unwrap();
    upload_values(&device, 0xad19_5832_0417_2635u64).unwrap();
    upload_values(&device, -173i32).unwrap();
    upload_values(&device, 1.5f32).unwrap();
    upload_values(&device, 1.5f64).unwrap();
    upload_values(&device, eunomia::F16::from_bits(0x3e00)).unwrap();
    upload_values(&device, eunomia::Bf16::from_bits(0x3fc0)).unwrap();
}

#[test]
fn storage_rejects_sizes_above_the_enabled_device_limit() {
    let Some(probe) = device_or_skip() else {
        return;
    };
    let mut limits = probe.limits();
    // A small real logical-device limit makes the rejected upload fixture
    // bounded independently of the adapter's physical allocation capacity.
    limits.max_buffer_size = 256;
    limits.max_storage_buffer_binding_size = 256;
    limits.max_uniform_buffer_binding_size = 256;
    drop(probe);
    let device = WgpuDevice::try_default_with_limits("storage allocation limit", limits).unwrap();
    assert_eq!(device.limits().max_buffer_size, 256);
    let host = [7u8; 257];
    for outcome in [
        device.alloc_uninitialized::<u8>(host.len()),
        device.alloc_zeroed::<u8>(host.len()),
        device.upload(&host),
    ] {
        match outcome {
            Err(HephaestusError::AllocationFailed { message }) => assert_eq!(
                message,
                "WGPU storage requires 260 padded bytes; enabled max_buffer_size=256"
            ),
            outcome => panic!("expected enabled-limit allocation rejection, got {outcome:?}"),
        }
    }
    let accepted = [13u8; 256];
    let buffer = device.upload(&accepted).unwrap();
    assert_eq!(device.download_owned(&buffer).unwrap(), accepted);
}
