use super::*;

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
fn device_local_copy_preserves_values_and_rejects_mismatch() {
    let Some(device) = device("device_local_copy_preserves_values_and_rejects_mismatch") else {
        return;
    };
    let host: Vec<f32> = (0..1027).map(|index| index as f32 * 0.25 - 17.0).collect();
    let source = device.upload(&host).expect("HIP source upload");
    let destination = device
        .alloc_zeroed::<f32>(host.len())
        .expect("HIP destination allocation");

    device
        .copy_buffer(&source, &destination)
        .expect("HIP device-local copy");
    let mut copied = vec![0.0_f32; host.len()];
    device
        .download(&destination, &mut copied)
        .expect("HIP copied-value download");
    assert_eq!(copied, host);

    let empty_source = device.upload(&[] as &[f32]).expect("empty HIP source");
    let empty_destination = device
        .alloc_zeroed::<f32>(0)
        .expect("empty HIP destination");
    device
        .copy_buffer(&empty_source, &empty_destination)
        .expect("empty HIP device-local copy");
    assert_eq!(empty_destination.len(), 0);

    let short = device
        .alloc_zeroed::<f32>(host.len() - 1)
        .expect("HIP short destination allocation");
    assert_length_mismatch(
        device.copy_buffer(&source, &short),
        source.len(),
        short.len(),
    );
}

#[test]
fn uninitialized_allocation_is_fully_overwritten_before_read() {
    let Some(device) = device("uninitialized_allocation_is_fully_overwritten_before_read") else {
        return;
    };
    let expected = [1.0_f32, -2.5, 3.25, 8.0];
    let buffer = device
        .alloc_uninitialized::<f32>(expected.len())
        .expect("uninitialized allocation");
    device
        .write_buffer(&buffer, &expected)
        .expect("full-buffer initialization");
    let mut actual = [0.0_f32; 4];
    device.download(&buffer, &mut actual).expect("download");
    assert_eq!(actual, expected);
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

#[test]
fn authored_kernel_streams_preserve_dispatch_copy_fill_and_grouped_values() {
    let Some(device) =
        device("authored_kernel_streams_preserve_dispatch_copy_fill_and_grouped_values")
    else {
        return;
    };

    let lhs = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("stream lhs upload");
    let rhs = device
        .upload(&[5.0_f32, 6.0, 7.0, 8.0])
        .expect("stream rhs upload");
    let output = device
        .alloc_zeroed::<f32>(4)
        .expect("stream output allocation");
    let grid = DispatchGrid::covering_domain([4, 1, 1], [64, 1, 1]).expect("stream grid");
    let prepared = device.prepare(&StreamAddKernel).expect("stream prepare");
    let bindings = [
        Binding::read(&lhs),
        Binding::read(&rhs),
        Binding::read_write(&output),
    ];
    let mut stream = device.stream().expect("stream open");
    stream
        .encode(&prepared, &bindings, &4, grid)
        .expect("stream dispatch");

    let copied = device
        .alloc_zeroed::<f32>(4)
        .expect("copy output allocation");
    let prefix = device
        .alloc_zeroed::<f32>(4)
        .expect("prefix output allocation");
    stream.copy(&output, &copied).expect("stream copy");
    stream
        .copy_prefix(&output, &prefix, 2)
        .expect("stream prefix copy");
    stream.fill_zero(&output).expect("stream fill");
    stream.submit().expect("stream submit");

    let mut output_values = [9.0_f32; 4];
    let mut copied_values = [0.0_f32; 4];
    let mut prefix_values = [0.0_f32; 4];
    device
        .download(&output, &mut output_values)
        .expect("stream output download");
    device
        .download(&copied, &mut copied_values)
        .expect("stream copy download");
    device
        .download(&prefix, &mut prefix_values)
        .expect("stream prefix download");
    assert_eq!(output_values, [0.0; 4]);
    assert_eq!(copied_values, [6.0, 8.0, 10.0, 12.0]);
    assert_eq!(prefix_values, [6.0, 8.0, 0.0, 0.0]);

    let grouped_prepared = device
        .prepare_grouped(&GroupedStreamAddKernel)
        .expect("grouped stream prepare");
    let grouped_output = device
        .alloc_zeroed::<f32>(4)
        .expect("grouped output allocation");
    let grouped_bindings = [
        GroupedBinding::read(0, 0, &lhs),
        GroupedBinding::read(0, 1, &rhs),
        GroupedBinding::read_write(0, 2, &grouped_output),
    ];
    let mut grouped_stream = device.grouped_stream().expect("grouped stream open");
    grouped_stream
        .encode_grouped_sequence("grouped-contract", |sequence| {
            sequence.encode_grouped(&grouped_prepared, &grouped_bindings, &4, grid)
        })
        .expect("grouped stream dispatch");
    grouped_stream
        .submit_grouped()
        .expect("grouped stream submit");

    let mut grouped_values = [0.0_f32; 4];
    device
        .download(&grouped_output, &mut grouped_values)
        .expect("grouped output download");
    assert_eq!(grouped_values, [6.0, 8.0, 10.0, 12.0]);
}
