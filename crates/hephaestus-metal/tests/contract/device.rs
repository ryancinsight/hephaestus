use super::*;

#[test]
fn metal_capabilities_match_the_selected_wgpu_context() {
    let Some(device) = device("metal_capabilities_match_the_selected_wgpu_context") else {
        return;
    };

    assert_eq!(device.device_limits(), device.wgpu_device().device_limits());
    for feature in [
        DeviceFeature::TimestampQuery,
        DeviceFeature::ShaderF64,
        DeviceFeature::ShaderF16,
        DeviceFeature::MappablePrimaryBuffers,
        DeviceFeature::ImmediateData,
    ] {
        assert_eq!(
            device.supports_device_feature(feature),
            device.wgpu_device().supports_device_feature(feature)
        );
    }
}

#[test]
fn metal_zero_device_acquisition_does_not_probe_or_fabricate_a_device() {
    let devices = MetalDevice::try_acquire_devices(
        "hephaestus-metal-zero-acquisition",
        0,
        DevicePreference::LowPower,
        &[],
        DeviceLimits {
            max_buffer_size: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroup_storage_size: 0,
            max_storage_buffers_per_shader_stage: None,
            max_buffers_and_acceleration_structures_per_shader_stage: None,
            max_immediate_size: 0,
        },
    )
    .expect("zero-device acquisition must terminate before adapter probing");

    assert!(devices.is_empty());
}

#[test]
fn metal_shared_acquisition_preserves_backend_features_and_limits() {
    let Some(reference) = device("metal_shared_acquisition_preserves_backend_features_and_limits")
    else {
        return;
    };
    let features = [
        DeviceFeature::TimestampQuery,
        DeviceFeature::ShaderF64,
        DeviceFeature::ShaderF16,
        DeviceFeature::MappablePrimaryBuffers,
        DeviceFeature::ImmediateData,
    ];
    let reference_wgpu = reference.wgpu_device();
    assert_eq!(
        reference_wgpu
            .adapter_info()
            .expect("Metal acquisition retains adapter metadata")
            .backend,
        wgpu::Backend::Metal
    );
    let adapter_features = reference_wgpu
        .adapter_features()
        .expect("Metal acquisition retains adapter features");
    assert!(adapter_features.contains(reference_wgpu.features()));
    let adapter_limits = reference_wgpu
        .adapter_limits()
        .expect("Metal acquisition retains adapter limits");
    assert!(adapter_limits.max_buffer_size >= reference_wgpu.limits().max_buffer_size);
    let required_limits = reference.device_limits();
    let acquired = MetalDevice::try_acquire_device(
        "hephaestus-metal-shared-acquisition",
        DevicePreference::HighPerformance,
        &features,
        required_limits,
    )
    .expect("shared acquisition must select the available Metal adapter");

    assert_eq!(acquired.backend_name(), "metal");
    assert_eq!(acquired.device_limits(), required_limits);
    assert_eq!(
        acquired
            .wgpu_device()
            .adapter_info()
            .expect("shared acquisition must retain adapter identity")
            .backend,
        wgpu::Backend::Metal
    );
    let requested_features = wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::SHADER_F64
        | wgpu::Features::SHADER_F16
        | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
        | wgpu::Features::IMMEDIATES;
    assert_eq!(
        acquired.wgpu_device().features(),
        acquired
            .wgpu_device()
            .adapter_features()
            .expect("shared acquisition must retain adapter features")
            & requested_features
    );

    let enumerated = MetalDevice::try_acquire_devices(
        "hephaestus-metal-shared-enumeration",
        1,
        DevicePreference::HighPerformance,
        &features,
        required_limits,
    )
    .expect("bounded acquisition must enumerate the available Metal adapter");
    assert_eq!(enumerated.len(), 1);
    assert_eq!(enumerated[0].backend_name(), "metal");
    assert_eq!(enumerated[0].device_limits(), required_limits);
    assert_eq!(
        enumerated[0]
            .wgpu_device()
            .adapter_info()
            .expect("enumeration must retain adapter identity")
            .backend,
        wgpu::Backend::Metal
    );
    assert_eq!(
        enumerated[0].wgpu_device().features(),
        enumerated[0]
            .wgpu_device()
            .adapter_features()
            .expect("enumeration must retain adapter features")
            & requested_features
    );

    let mut impossible_limits = required_limits;
    impossible_limits.max_buffer_size = u64::MAX;
    match MetalDevice::try_acquire_device(
        "hephaestus-metal-unsatisfiable-acquisition",
        DevicePreference::HighPerformance,
        &features,
        impossible_limits,
    ) {
        Err(HephaestusError::AdapterUnavailable { .. }) => {}
        other => panic!("unsatisfiable Metal limits must reject acquisition, got {other:?}"),
    }
}

#[test]
fn grouped_metal_stream_preserves_grouped_output_and_sequence_order() {
    let Some(device) = device("grouped_metal_stream_preserves_grouped_output_and_sequence_order")
    else {
        return;
    };

    let left = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let grid = DispatchGrid::covering_domain([4, 1, 1], [64, 1, 1]).unwrap();
    let bindings = [
        GroupedBinding::read(0, 0, &left),
        GroupedBinding::read(1, 0, &right),
        GroupedBinding::read_write(1, 1, &output),
    ];
    let params = GroupedParams {
        len: 4,
        addend: 0.5,
    };

    let mut stream = device.grouped_stream().unwrap();
    stream
        .encode_grouped(&prepared, &bindings, &params, grid)
        .unwrap();
    stream
        .encode_grouped_sequence("metal-grouped-sequence", |sequence| {
            sequence.encode_grouped(&prepared, &bindings, &params, grid)
        })
        .unwrap();
    stream.submit_grouped().unwrap();

    let mut got = [0.0_f32; 4];
    device.download(&output, &mut got).unwrap();
    assert_eq!(got, [11.5, 22.5, 33.5, 44.5]);
}

#[test]
fn fluent_linalg_traits_match_value_contracts() {
    let Some(device) = device("fluent_linalg_traits_match_value_contracts") else {
        return;
    };

    let matrix = device.upload(&[4.0_f32, 1.0, 1.0, 3.0]).unwrap();
    let matrix_layout = Layout::c_contiguous([2, 2]).unwrap();
    let operand = StridedOperand {
        buffer: &matrix,
        layout: &matrix_layout,
    };

    let product = operand.matmul(&device, &operand).unwrap();
    let mut got_product = [0.0_f32; 4];
    device.download(&product, &mut got_product).unwrap();
    assert_eq!(got_product, [17.0, 7.0, 7.0, 10.0]);

    let l1 = operand.norm_l1(&device).unwrap();
    let l2 = operand.norm_l2(&device).unwrap();
    let max = operand.norm_max(&device).unwrap();
    let trace = operand.trace(&device).unwrap();
    let det = operand.det(&device).unwrap();
    let mut got_scalar = [0.0_f32];
    device.download(&l1, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [9.0]);
    device.download(&l2, &mut got_scalar).unwrap();
    assert!((got_scalar[0] - 27.0_f32.sqrt()).abs() <= 8.0 * f32::EPSILON);
    device.download(&max, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [4.0]);
    device.download(&trace, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [7.0]);
    device.download(&det, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [11.0]);
    assert_eq!(operand.rank(&device).unwrap(), 2);
    assert_eq!(operand.rank_with_tolerance(&device, 1.0e-6).unwrap(), 2);

    let power = operand.matpow(&device, 2).unwrap();
    let mut got_power = [0.0_f32; 4];
    device.download(&power, &mut got_power).unwrap();
    assert_eq!(got_power, got_product);

    let rhs = device.upload(&[5.0_f32, 7.0]).unwrap();
    let solution = operand.solve(&device, &rhs).unwrap();
    let mut got_solution = [0.0_f32; 2];
    device.download(&solution, &mut got_solution).unwrap();
    assert!((got_solution[0] - 8.0 / 11.0).abs() <= 8.0 * f32::EPSILON);
    assert!((got_solution[1] - 23.0 / 11.0).abs() <= 8.0 * f32::EPSILON);

    let inverse = operand.inv(&device).unwrap();
    let pseudoinverse = operand.pinv(&device).unwrap();
    let mut got_inverse = [0.0_f32; 4];
    let mut got_pseudoinverse = [0.0_f32; 4];
    device.download(&inverse, &mut got_inverse).unwrap();
    device
        .download(&pseudoinverse, &mut got_pseudoinverse)
        .unwrap();
    let expected_inverse = [3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0];
    for (got, expected) in got_inverse.iter().zip(expected_inverse) {
        assert!((got - expected).abs() <= 16.0 * f32::EPSILON);
    }
    for (got, expected) in got_pseudoinverse.iter().zip(expected_inverse) {
        assert!((got - expected).abs() <= 16.0 * f32::EPSILON);
    }

    let diagonal = device.upload(&[1.0_f32, 0.0, 0.0, 2.0]).unwrap();
    let diagonal_operand = StridedOperand {
        buffer: &diagonal,
        layout: &matrix_layout,
    };
    let exponential = diagonal_operand.matexp(&device).unwrap();
    let mut got_exponential = [0.0_f32; 4];
    device.download(&exponential, &mut got_exponential).unwrap();
    assert!((got_exponential[0] - 1.0_f32.exp()).abs() <= 16.0 * f32::EPSILON);
    assert!((got_exponential[3] - 2.0_f32.exp()).abs() <= 16.0 * f32::EPSILON);
}
