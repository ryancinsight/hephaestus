use super::fixtures::{ScaleKernel, ScaleParams, try_device};
use hephaestus_core::{
    Binding, CommandStream, ComputeDevice, DispatchGrid, HephaestusError, KernelDevice,
};

pub(super) fn command_stream_dispatches_prepared_wgsl_kernel() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping command stream test");
        return;
    };
    let input = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare(&ScaleKernel).unwrap();

    device
        .dispatch(
            &prepared,
            &[Binding::read(&input), Binding::read_write(&output)],
            &ScaleParams {
                len: 4,
                factor: 2.5,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [2.5, 5.0, 7.5, 10.0]);
}

pub(super) fn command_stream_preserves_order_for_fill_copy_and_dispatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping command stream order test");
        return;
    };
    let input = device.upload(&[3.0f32, 4.0, 5.0, 6.0]).unwrap();
    let scratch = device.alloc_zeroed::<f32>(4).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare(&ScaleKernel).unwrap();

    let mut stream = device.stream().unwrap();
    stream.fill_zero(&scratch).unwrap();
    stream.copy(&input, &scratch).unwrap();
    stream
        .encode(
            &prepared,
            &[Binding::read(&scratch), Binding::read_write(&output)],
            &ScaleParams {
                len: 4,
                factor: 3.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();
    stream.submit().unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [9.0, 12.0, 15.0, 18.0]);
}

pub(super) fn command_stream_rejects_binding_contract_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping binding contract test");
        return;
    };
    let input = device.upload(&[1.0f32, 2.0]).unwrap();
    let prepared = device.prepare(&ScaleKernel).unwrap();

    let err = device
        .dispatch(
            &prepared,
            &[Binding::read(&input)],
            &ScaleParams {
                len: 2,
                factor: 1.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap_err();

    match err {
        HephaestusError::DispatchFailed { message } => {
            assert!(message.contains("declares 2 storage bindings, got 1"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

pub(super) fn command_stream_copy_rejects_length_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping copy contract test");
        return;
    };
    let src = device.upload(&[1u32, 2, 3]).unwrap();
    let dst = device.alloc_zeroed::<u32>(2).unwrap();
    let mut stream = device.stream().unwrap();

    let err = stream.copy(&src, &dst).unwrap_err();

    assert!(matches!(
        err,
        HephaestusError::LengthMismatch {
            host_len: 3,
            device_len: 2
        }
    ));
}

pub(super) fn command_stream_copy_prefix_preserves_the_destination_suffix() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping command stream prefix-copy test");
        return;
    };
    let src = device.upload(&[3u32, 5, 7, 11]).unwrap();
    let dst = device.alloc_zeroed::<u32>(4).unwrap();
    let mut stream = device.stream().unwrap();
    stream.copy_prefix(&src, &dst, 2).unwrap();
    stream.submit().unwrap();

    let mut host = [0u32; 4];
    device.download(&dst, &mut host).unwrap();
    assert_eq!(host, [3, 5, 0, 0]);
}
