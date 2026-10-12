use super::*;

#[test]
fn upload_download_roundtrip_f32() {
    let Some(dev) = device("upload_download_roundtrip_f32") else {
        return;
    };
    assert_eq!(dev.backend_name(), "cuda");
    let host = vec![1.0f32, 2.0, -3.5, 4.25, 0.0, 1024.5];
    let buf = dev.upload(&host).expect("upload");
    assert_eq!(buf.len(), host.len());
    let mut out = vec![0.0f32; host.len()];
    dev.download(&buf, &mut out).expect("download");
    assert_eq!(out, host, "round-trip must be the identity");
}

#[test]
fn device_local_copy_preserves_values_and_rejects_mismatch() {
    let Some(dev) = device("device_local_copy_preserves_values_and_rejects_mismatch") else {
        return;
    };
    let host: Vec<f32> = (0..1027).map(|index| index as f32 * 0.25 - 17.0).collect();
    let source = dev.upload(&host).expect("upload source");
    let destination = dev
        .alloc_zeroed::<f32>(host.len())
        .expect("allocate destination");

    dev.copy_buffer(&source, &destination)
        .expect("device-local copy");
    let mut copied = vec![0.0_f32; host.len()];
    dev.download(&destination, &mut copied)
        .expect("download copied values");
    assert_eq!(copied, host);

    let empty_source = dev.upload::<f32>(&[]).expect("upload empty source");
    let empty_destination = dev
        .alloc_zeroed::<f32>(0)
        .expect("allocate empty destination");
    dev.copy_buffer(&empty_source, &empty_destination)
        .expect("empty device-local copy");

    let short = dev
        .alloc_zeroed::<f32>(host.len() - 1)
        .expect("allocate short destination");
    assert_length_mismatch(dev.copy_buffer(&source, &short), source.len(), short.len());
}

#[test]
fn uninitialized_allocation_is_fully_overwritten_before_read() {
    let Some(dev) = device("uninitialized_allocation_is_fully_overwritten_before_read") else {
        return;
    };
    let expected = [1.0_f32, -2.5, 3.25, 8.0];
    let buffer = dev
        .alloc_uninitialized::<f32>(expected.len())
        .expect("uninitialized allocation");
    dev.write_buffer(&buffer, &expected)
        .expect("full-buffer initialization");
    let mut actual = [0.0_f32; 4];
    dev.download(&buffer, &mut actual).expect("download");
    assert_eq!(actual, expected);
}

#[test]
fn test_placement_aware_allocation() {
    let Some(dev) = device("test_placement_aware_allocation") else {
        return;
    };
    use themis::{MemoryTier, PlacementHint};

    // CUDA primary buffers use non-managed `cuMemAlloc_v2` device memory even
    // when a host-visible placement hint is supplied.
    let hint = PlacementHint::Tier(MemoryTier::HostPinned);
    let buf1 = dev.alloc_zeroed_with_hint::<f32>(128, hint).unwrap();
    assert_eq!(buf1.len(), 128);
    assert_eq!(buf1.tier(), MemoryTier::Device);

    let host = vec![1.5f32; 128];
    let buf2 = dev.upload_with_hint(&host, hint).unwrap();
    assert_eq!(buf2.len(), 128);
    assert_eq!(buf2.tier(), MemoryTier::Device);

    // Test Dram / unified host memory hints normalize to the implemented
    // non-managed device tier.
    let hint_dram = PlacementHint::Tier(MemoryTier::Dram);
    let buf3 = dev.alloc_zeroed_with_hint::<f32>(128, hint_dram).unwrap();
    assert_eq!(buf3.tier(), MemoryTier::Device);

    let registers =
        dev.alloc_zeroed_with_hint::<f32>(128, PlacementHint::Tier(MemoryTier::Registers));
    match registers {
        Err(HephaestusError::AllocationFailed { message }) => assert_eq!(
            message,
            "CUDA primary buffers cannot be allocated from budget-only tier Registers"
        ),
        other => panic!("expected budget-only tier rejection, got {other:?}"),
    }

    // Test default non-hinted delegates
    let buf4 = dev.alloc_zeroed::<f32>(128).unwrap();
    assert_eq!(buf4.tier(), MemoryTier::Device);
}

#[test]
fn upload_download_roundtrip_i32() {
    let Some(dev) = device("upload_download_roundtrip_i32") else {
        return;
    };
    let host: Vec<i32> = (-4..=4).collect();
    let buf = dev.upload(&host).expect("upload");
    let mut out = vec![0i32; host.len()];
    dev.download(&buf, &mut out).expect("download");
    assert_eq!(out, host);
}

#[test]
fn empty_buffer_roundtrips() {
    let Some(dev) = device("empty_buffer_roundtrips") else {
        return;
    };
    let buf = dev.upload::<f32>(&[]).expect("upload empty");
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
    let mut out: Vec<f32> = Vec::new();
    dev.download(&buf, &mut out).expect("download empty");
}

#[test]
fn download_length_mismatch_rejected() {
    let Some(dev) = device("download_length_mismatch_rejected") else {
        return;
    };
    let buf = dev.upload(&[1.0f32, 2.0]).expect("upload");
    let mut out = vec![0.0f32; 5];
    let err = dev
        .download(&buf, &mut out)
        .expect_err("length mismatch must be rejected");
    assert!(
        matches!(
            err,
            HephaestusError::LengthMismatch {
                host_len: 5,
                device_len: 2,
            }
        ),
        "expected LengthMismatch{{5, 2}}, got {err:?}"
    );
}

#[test]
fn write_buffer_overwrites_existing_data() {
    let Some(dev) = device("write_buffer_overwrites_existing_data") else {
        return;
    };

    // Upload initial data.
    let initial = vec![1.0f32, 2.0, 3.0, 4.0];
    let buf = dev.upload(&initial).unwrap();

    // Overwrite with new data via write_buffer.
    let updated = vec![10.0f32, 20.0, 30.0, 40.0];
    dev.write_buffer(&buf, &updated).unwrap();

    // Download and verify the overwritten data.
    let mut got = vec![0.0f32; 4];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, updated);
}

#[test]
fn write_buffer_rejects_length_mismatch() {
    let Some(dev) = device("write_buffer_rejects_length_mismatch") else {
        return;
    };

    let buf = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let wrong_len = vec![1.0f32, 2.0]; // len 2, buffer len 3
    assert_length_mismatch(dev.write_buffer(&buf, &wrong_len), 2, 3);
}

#[test]
fn write_buffer_empty_is_noop() {
    let Some(dev) = device("write_buffer_empty_is_noop") else {
        return;
    };

    let buf = dev.upload::<f32>(&[]).unwrap();
    dev.write_buffer(&buf, &[] as &[f32]).unwrap();
    assert_eq!(buf.len(), 0);
}

#[test]
fn write_buffer_integer_types() {
    let Some(dev) = device("write_buffer_integer_types") else {
        return;
    };

    let buf = dev.upload(&[0i32, 0, 0]).unwrap();
    let data = vec![42i32, -7, 100];
    dev.write_buffer(&buf, &data).unwrap();

    let mut got = vec![0i32; 3];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, data);
}

#[test]
fn write_sub_buffer_overwrites_only_requested_range() {
    let Some(dev) = device("write_sub_buffer_overwrites_only_requested_range") else {
        return;
    };

    let buf = dev.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    dev.write_sub_buffer(&buf, 1, &[20.0f32, 30.0]).unwrap();

    let mut got = [0.0f32; 4];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, [1.0, 20.0, 30.0, 4.0]);
}

#[test]
fn write_sub_buffer_rejects_out_of_range_write() {
    let Some(dev) = device("write_sub_buffer_rejects_out_of_range_write") else {
        return;
    };

    let buf = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(dev.write_sub_buffer(&buf, 2, &[4.0f32, 5.0]), 4, 3);
}

#[test]
fn write_sub_buffer_empty_tail_write_is_noop() {
    let Some(dev) = device("write_sub_buffer_empty_tail_write_is_noop") else {
        return;
    };

    let buf = dev.upload(&[9i32, 8, 7]).unwrap();
    dev.write_sub_buffer(&buf, 3, &[] as &[i32]).unwrap();

    let mut got = [0i32; 3];
    dev.download(&buf, &mut got).unwrap();
    assert_eq!(got, [9, 8, 7]);
}

/// The regression guard for the defect itself.
///
/// Neither value clause below discriminates it: the old field was a free
/// reading *stored at acquisition*, so it was stale rather than moving, and
/// `buffer_limit_is_stable_across_allocations_and_free_memory_tracks_them`
/// passes on the defective code. `free <= max_buffer_size` also holds when
/// the limit *is* a free reading. What distinguishes the two is which half of
/// `cuMemGetInfo_v2` reaches `DeviceLimits`, so the source is the oracle —
/// and this clause needs no device, so it guards on every runner.
#[test]
fn the_buffer_limit_is_built_from_total_device_memory_not_the_free_reading() {
    let source = include_str!("../../src/infrastructure/device/properties.rs");
    let body = source
        .split_once("fn query_device_limits(")
        .map(|(_, tail)| tail)
        .and_then(|tail| tail.split_once("\npub(super) fn query_device_features("))
        .map(|(body, _)| body)
        .expect("query_device_limits must be present");

    assert!(
        body.contains("let (_, total_bytes) = current_memory_info(driver)?;"),
        "query_device_limits must take the total half of current_memory_info"
    );
    assert!(
        !body.contains("free_bytes"),
        "no free-memory reading may reach DeviceLimits; it is a runtime query"
    );
}

/// `max_buffer_size` is the device capacity — it does not move when memory is
/// allocated — while the free-memory query does.
#[test]
fn buffer_limit_is_stable_across_allocations_and_free_memory_tracks_them() {
    let Some(dev) = device("buffer_limit_is_stable_across_allocations") else {
        return;
    };
    let limits_before = dev.device_limits();
    let free_before = dev.free_memory_bytes().expect("free-memory query");
    // 64 MiB: large enough that the driver must back it with new device pages
    // rather than serve it from a pool the previous tests already touched.
    let _held = dev
        .alloc_zeroed::<f32>(16 << 20)
        .expect("64 MiB device buffer");
    let free_after = dev.free_memory_bytes().expect("free-memory query");
    assert_eq!(
        dev.device_limits(),
        limits_before,
        "the limit moved with an allocation"
    );
    assert!(
        free_after < free_before,
        "free memory did not fall across a 64 MiB allocation: {free_before} -> {free_after}"
    );
}
