use super::fixtures::{GroupedAddKernel, GroupedParams, ScaleKernel, try_device};
use hephaestus_core::{
    Access, ComputeDevice, DispatchGrid, GroupedBinding, GroupedCommandStream, GroupedKernelDevice,
    GroupedKernelSequence, HephaestusError, KernelInterface,
};

pub(super) fn grouped_command_stream_dispatches_multi_group_wgsl_kernel() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping grouped command stream test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    device
        .dispatch_grouped(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 4,
                addend: 0.5,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [11.5, 22.5, 33.5, 44.5]);
}

pub(super) fn grouped_sequence_preserves_order_in_one_wgpu_pass() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping grouped sequence test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let scratch = device.alloc_zeroed::<f32>(4).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    let mut stream = device.grouped_stream().unwrap();
    stream
        .encode_grouped_sequence("hephaestus-wgpu-grouped-sequence", |sequence| {
            sequence.encode_grouped(
                &prepared,
                &[
                    GroupedBinding::read(0, 0, &left),
                    GroupedBinding::read(1, 0, &right),
                    GroupedBinding::read_write(1, 1, &scratch),
                ],
                &GroupedParams {
                    len: 4,
                    addend: 0.5,
                },
                DispatchGrid::new(1, 1, 1),
            )?;
            sequence.encode_grouped(
                &prepared,
                &[
                    GroupedBinding::read(0, 0, &scratch),
                    GroupedBinding::read(1, 0, &right),
                    GroupedBinding::read_write(1, 1, &output),
                ],
                &GroupedParams {
                    len: 4,
                    addend: 1.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
        })
        .unwrap();
    stream.submit_grouped().unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [22.5, 43.5, 64.5, 85.5]);
}

pub(super) fn grouped_command_stream_rejects_group_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping grouped mismatch test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    let err = device
        .dispatch_grouped(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(0, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 1,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap_err();

    match err {
        HephaestusError::DispatchFailed { message } => {
            assert!(message.contains("declared group 1 binding 0"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

pub(super) fn kernel_interface_declares_wgpu_parameter_binding_after_storage() {
    assert_eq!(ScaleKernel::BINDINGS.len(), 2);
    assert_eq!(ScaleKernel::BINDINGS[0].access, Access::ReadOnly);
    assert_eq!(ScaleKernel::BINDINGS[1].access, Access::ReadWrite);
}
