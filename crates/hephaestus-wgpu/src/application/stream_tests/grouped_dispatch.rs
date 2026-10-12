use super::*;

pub(crate) fn grouped_command_stream_dispatches_multi_group_wgsl_kernel() {
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

pub(crate) fn grouped_sequence_preserves_order_in_one_wgpu_pass() {
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

pub(crate) fn bound_grouped_sequence_reuses_fixed_resources() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound grouped stream test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let scratch = device.alloc_zeroed::<f32>(4).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let first = device
        .bind_grouped_dispatch(
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
        )
        .unwrap();
    let second = device
        .bind_grouped_dispatch(
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
        .unwrap();

    for (input, expected) in [
        ([1.0, 2.0, 3.0, 4.0], [22.5, 43.5, 64.5, 85.5]),
        ([2.0, 4.0, 6.0, 8.0], [23.5, 45.5, 67.5, 89.5]),
    ] {
        device.write_buffer(&left, &input).unwrap();
        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped_sequence("hephaestus-wgpu-bound-grouped", |sequence| {
                first.encode_in_sequence(sequence)?;
                second.encode_in_sequence(sequence)
            })
            .unwrap();
        stream.submit_grouped().unwrap();
        let mut host = [0.0f32; 4];
        device.download(&output, &mut host).unwrap();
        assert_eq!(host, expected);
    }

    let other = try_device().expect("second WGPU device");
    let error = other
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 4,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .expect_err("bound grouped preparation must preserve device provenance");
    assert!(error.to_string().contains("different device"));
}

pub(crate) fn bound_grouped_dispatch_updates_retained_parameters() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound parameter-update test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let mut bound = device
        .bind_grouped_dispatch(
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

    let execute = |bound: &WgpuBoundGroupedDispatch<GroupedAddKernel>| {
        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped_sequence("hephaestus-wgpu-bound-parameter-update", |sequence| {
                bound.encode_in_sequence(sequence)
            })
            .unwrap();
        stream.submit_grouped().unwrap();
    };
    execute(&bound);
    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [11.5, 22.5, 33.5, 44.5]);

    bound
        .update_params(
            &device,
            &GroupedParams {
                len: 2,
                addend: -1.0,
            },
        )
        .unwrap();
    execute(&bound);
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [10.0, 21.0, 33.5, 44.5]);
}

pub(crate) fn bound_grouped_parameter_update_rejects_foreign_device() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping foreign parameter-update test");
        return;
    };
    let Some(other) = try_device() else {
        eprintln!("No second WGPU device available; skipping foreign parameter-update test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let mut bound = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 1,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    let error = bound
        .update_params(
            &other,
            &GroupedParams {
                len: 1,
                addend: 9.0,
            },
        )
        .expect_err("parameter update must reject a foreign device");
    match error {
        HephaestusError::DispatchFailed { message } => assert_eq!(
            message,
            "prepared WGPU bound grouped dispatch belongs to a different device"
        ),
        other => panic!("unexpected error: {other:?}"),
    }
}

pub(crate) fn bound_grouped_dispatch_rejects_invalid_parameter_size() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound parameter-size test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let oversized = WgpuGroupedPrepared::<OversizedGroupedKernel> {
        pipeline: prepared.pipeline.clone(),
        bind_group_layouts: prepared.bind_group_layouts.clone(),
        parameter_group: prepared.parameter_group,
        parameter_binding: prepared.parameter_binding,
        owner: prepared.owner.clone(),
        label: OversizedGroupedKernel::LABEL,
        marker: PhantomData,
    };

    let error = device
        .bind_grouped_dispatch(
            &oversized,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &OVERSIZED_PARAMETERS,
            DispatchGrid::new(1, 1, 1),
        )
        .expect_err("oversized retained parameters must fail before WGPU allocation");

    match error {
        HephaestusError::AllocationFailed { message } => {
            assert!(message.contains("65540 bytes"));
            assert!(message.contains("max_uniform_buffer_binding_size=65536"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

pub(crate) fn bound_grouped_sequence_rejects_foreign_device() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping foreign bound-sequence test");
        return;
    };
    let Some(other) = try_device() else {
        eprintln!("No second WGPU device available; skipping foreign bound-sequence test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let bound = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 1,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();
    let mut stream = other.grouped_stream().unwrap();

    let error = stream
        .encode_grouped_sequence("foreign-bound-grouped-sequence", |sequence| {
            bound.encode_in_sequence(sequence)
        })
        .expect_err("bound dispatch must reject a foreign sequence");

    assert_eq!(
        error.to_string(),
        "kernel dispatch failed: prepared WGPU bound grouped dispatch belongs to a different device"
    );
    let mut host = [f32::NAN; 1];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [0.0]);
}

pub(crate) fn bound_grouped_dispatch_rejects_group_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound grouped mismatch test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    let error = device
        .bind_grouped_dispatch(
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
        .expect_err("bound grouped preparation must validate binding slots");

    assert_eq!(
        error.to_string(),
        "kernel dispatch failed: hephaestus-wgpu-grouped-add: binding 1 declared group 1 binding 0, bound group 0 binding 0"
    );
}

pub(crate) fn grouped_command_stream_rejects_group_mismatch() {
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
