use super::*;

pub(crate) fn upload_download_round_trips_values() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host: Vec<f32> = (0..1027).map(|i| i as f32 * 0.5 - 100.0).collect();
    let buffer = device.upload(&host).unwrap();
    assert_eq!(buffer.len(), host.len());

    let mut out = vec![0.0f32; host.len()];
    device.download(&buffer, &mut out).unwrap();
    assert_eq!(out, host);
}

pub(crate) fn device_local_copy_preserves_values_and_rejects_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host: Vec<f32> = (0..1027).map(|index| index as f32 * 0.25 - 17.0).collect();
    let source = device.upload(&host).unwrap();
    let destination = device.alloc_zeroed::<f32>(host.len()).unwrap();

    device.copy_buffer(&source, &destination).unwrap();
    let mut copied = vec![0.0_f32; host.len()];
    device.download(&destination, &mut copied).unwrap();
    assert_eq!(copied, host);

    let short = device.alloc_zeroed::<f32>(host.len() - 1).unwrap();
    assert_length_mismatch(
        device.copy_buffer(&source, &short),
        source.len(),
        short.len(),
    );
}

pub(crate) fn uninitialized_allocation_is_fully_overwritten_before_read() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let expected = [1.0_f32, -2.5, 3.25, 8.0];
    let buffer = device.alloc_uninitialized::<f32>(expected.len()).unwrap();
    device.write_buffer(&buffer, &expected).unwrap();
    let mut actual = [0.0_f32; 4];
    device.download(&buffer, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

pub(crate) fn odd_u16_storage_preserves_logical_values_when_device_exists() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let uploaded: Vec<u16> = (0..27).map(|index| index * 257 + 1).collect();
    let upload_buffer = device.upload(&uploaded).expect("odd u16 upload");
    assert_eq!(upload_buffer.len(), uploaded.len());
    let mut uploaded_readback = vec![0_u16; uploaded.len()];
    device
        .download(&upload_buffer, &mut uploaded_readback)
        .expect("odd u16 upload readback");
    assert_eq!(uploaded_readback, uploaded);

    let written: Vec<u16> = uploaded.iter().map(|value| value ^ 0x55AA).collect();
    let allocated = device
        .alloc_zeroed::<u16>(written.len())
        .expect("odd u16 allocation");
    device
        .write_buffer(&allocated, &written)
        .expect("odd u16 full-buffer write");
    let mut written_readback = vec![0_u16; written.len()];
    device
        .download(&allocated, &mut written_readback)
        .expect("odd u16 full-buffer readback");
    assert_eq!(written_readback, written);
}

pub(crate) fn test_placement_aware_allocation() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use themis::{MemoryTier, PlacementHint};

    let host: Vec<f32> = (0..128).map(|i| i as f32 * 0.25 - 7.0).collect();

    // WGPU exposes device buffers and transient mapped transfer buffers, but
    // does not guarantee host-pinned or host-DRAM placement. Unsupported tier
    // requests fail at the boundary instead of recording a false tier.
    for tier in [MemoryTier::HostPinned, MemoryTier::Dram] {
        let hint = PlacementHint::Tier(tier);
        let expected =
            format!("WGPU cannot guarantee requested memory tier {tier:?}; use Device placement");
        for error in [
            device
                .upload_with_hint(&host, hint)
                .expect_err("unsupported upload placement must fail"),
            device
                .alloc_zeroed_with_hint::<f32>(128, hint)
                .expect_err("unsupported allocation placement must fail"),
        ] {
            match error {
                HephaestusError::AllocationFailed { message } => {
                    assert_eq!(message, expected)
                }
                other => panic!("expected allocation failure, got {other:?}"),
            }
        }
    }

    // Default (non-hinted) allocation lands on Device and round-trips data.
    let buf_default = device.upload(&host).unwrap();
    assert_eq!(buf_default.tier(), MemoryTier::Device);
    let mut default_out = vec![0.0f32; 128];
    device.download(&buf_default, &mut default_out).unwrap();
    assert_eq!(default_out, host, "Device upload must preserve data");
}

pub(crate) fn download_rejects_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let buffer = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let mut out = vec![0.0f32; 2];
    assert_length_mismatch(device.download(&buffer, &mut out), 2, 3);
}

pub(crate) fn write_buffer_overwrites_existing_data() {
    let Some(device) = device_or_skip() else {
        return;
    };

    // Upload initial data.
    let initial = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = device.upload(&initial).unwrap();

    // Overwrite with new data via write_buffer.
    let updated = vec![10.0f32, 20.0, 30.0, 40.0];
    device.write_buffer(&buf, &updated).unwrap();

    // Download and verify the overwritten data.
    let mut got = vec![0.0f32; 4];
    device.download(&buf, &mut got).unwrap();
    assert_eq!(got, updated);
}

pub(crate) fn write_buffer_rejects_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let wrong_len = vec![1.0f32, 2.0]; // len 2, buffer len 3
    let result: hephaestus_wgpu::Result<()> = device.write_buffer(&buf, &wrong_len);
    assert_length_mismatch(result, 2, 3);
}

pub(crate) fn write_buffer_empty_is_noop() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload::<f32>(&[]).unwrap();
    device.write_buffer(&buf, &[] as &[f32]).unwrap();
    assert_eq!(buf.len(), 0);
}

pub(crate) fn write_buffer_integer_types() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload(&[0i32, 0, 0]).unwrap();
    let data = vec![42i32, -7, 100];
    device.write_buffer(&buf, &data).unwrap();

    let mut got = vec![0i32; 3];
    device.download(&buf, &mut got).unwrap();
    assert_eq!(got, data);
}

pub(crate) fn write_sub_buffer_overwrites_only_requested_range() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    device.write_sub_buffer(&buf, 1, &[20.0f32, 30.0]).unwrap();

    let mut got = [0.0f32; 4];
    device.download(&buf, &mut got).unwrap();
    assert_eq!(got, [1.0, 20.0, 30.0, 4.0]);
}

pub(crate) fn write_sub_buffer_rejects_out_of_range_write() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(device.write_sub_buffer(&buf, 2, &[4.0f32, 5.0]), 4, 3);
}

pub(crate) fn write_sub_buffer_empty_tail_write_is_noop() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let buf = device.upload(&[9i32, 8, 7]).unwrap();
    device.write_sub_buffer(&buf, 3, &[] as &[i32]).unwrap();

    let mut got = [0i32; 3];
    device.download(&buf, &mut got).unwrap();
    assert_eq!(got, [9, 8, 7]);
}
