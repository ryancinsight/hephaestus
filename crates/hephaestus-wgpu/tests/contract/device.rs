use super::*;

pub(crate) fn acquisition_reports_themis_topology_from_adapter() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let topology = device
        .topology()
        .expect("acquisition path must capture a topology snapshot");

    // A reported warp width is a real power-of-two lane count.
    assert!(
        topology
            .warp_width()
            .is_none_or(|width| width.get().is_power_of_two())
    );

    // Capacities WebGPU cannot introspect are None by type, never
    // fabricated (formerly a pinned zero-sentinel convention).
    assert_eq!(topology.compute_units(), None);
    assert_eq!(topology.registers_per_unit(), None);
    assert_eq!(topology.shared_mem_per_unit_bytes(), None);

    // The Arc-wrapping constructor has no adapter and reports none.
    let wrapped = WgpuDevice::new(device.device().clone(), device.queue().clone());
    assert_eq!(
        wrapped.topology().map(|topology| topology.compute_units()),
        None
    );
}

pub(crate) fn fdtd_3d_provider_matches_sequential_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let params = Fdtd3dParams::new(
        5,
        5,
        5,
        Length::from_unit::<Meter>(1.0),
        Length::from_unit::<Meter>(1.0),
        Length::from_unit::<Meter>(1.0),
        Time::from_unit::<Second>(0.01),
    )
    .expect("valid FDTD grid");
    let len = params.storage_len().expect("valid FDTD storage");
    let mut pressure = vec![0.0_f32; len];
    let velocity = vec![FdtdVelocity::zeroed(); len];
    let medium = vec![
        FdtdMedium::new(
            MassDensity::from_unit::<KilogramPerCubicMeter>(2.0),
            Velocity::from_unit::<MeterPerSecond>(3.0),
        )
        .expect("valid medium");
        len
    ];
    for z in 0..5 {
        for y in 0..5 {
            for x in 0..5 {
                pressure[x + y * 5 + z * 25] = (x * 3 + y * 2 + z) as f32;
            }
        }
    }

    let mut expected_velocity = velocity.clone();
    let mut expected_pressure = pressure.clone();
    for _ in 0..2 {
        let previous_pressure = expected_pressure.clone();
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let index = x + y * 5 + z * 25;
                    if x == 0 || x == 4 || y == 0 || y == 4 || z == 0 || z == 4 {
                        expected_velocity[index] = FdtdVelocity::zeroed();
                        continue;
                    }
                    let gradient = [
                        (previous_pressure[index + 1] - previous_pressure[index - 1]) * 0.5,
                        (previous_pressure[index + 5] - previous_pressure[index - 5]) * 0.5,
                        (previous_pressure[index + 25] - previous_pressure[index - 25]) * 0.5,
                    ];
                    let previous = expected_velocity[index];
                    expected_velocity[index] = FdtdVelocity {
                        components: [
                            previous.components[0] - 0.01 * gradient[0] / 2.0,
                            previous.components[1] - 0.01 * gradient[1] / 2.0,
                            previous.components[2] - 0.01 * gradient[2] / 2.0,
                        ],
                        padding: 0.0,
                    };
                }
            }
        }
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    let index = x + y * 5 + z * 25;
                    if x == 0 || x == 4 || y == 0 || y == 4 || z == 0 || z == 4 {
                        expected_pressure[index] = 0.0;
                        continue;
                    }
                    let divergence = (expected_velocity[index + 1].components[0]
                        - expected_velocity[index - 1].components[0])
                        * 0.5
                        + (expected_velocity[index + 5].components[1]
                            - expected_velocity[index - 5].components[1])
                            * 0.5
                        + (expected_velocity[index + 25].components[2]
                            - expected_velocity[index - 25].components[2])
                            * 0.5;
                    expected_pressure[index] =
                        previous_pressure[index] - 0.01 * 2.0 * 9.0 * divergence;
                }
            }
        }
    }

    let pressure_buffer = device.upload(&pressure).expect("upload pressure");
    let velocity_buffer = device.upload(&velocity).expect("upload velocity");
    let medium_buffer = device.upload(&medium).expect("upload medium");
    let kernel = WgpuFdtd3dOps
        .prepare_fdtd_3d(&device)
        .expect("compile FDTD provider kernels");
    for _ in 0..2 {
        WgpuFdtd3dOps
            .step_fdtd_3d(
                &device,
                &kernel,
                &pressure_buffer,
                &velocity_buffer,
                &medium_buffer,
                &params,
            )
            .expect("dispatch FDTD provider step");
    }
    device.synchronize().expect("synchronize FDTD step");

    let mut actual_pressure = vec![0.0_f32; len];
    let mut actual_velocity = vec![FdtdVelocity::zeroed(); len];
    device
        .download(&pressure_buffer, &mut actual_pressure)
        .expect("download pressure");
    device
        .download(&velocity_buffer, &mut actual_velocity)
        .expect("download velocity");
    let tolerance = 32.0 * f32::EPSILON;
    for (actual, expected) in actual_pressure.iter().zip(&expected_pressure) {
        assert_close(*actual, *expected, tolerance);
    }
    for (actual, expected) in actual_velocity.iter().zip(&expected_velocity) {
        for (actual, expected) in actual.components.iter().zip(expected.components) {
            assert_close(*actual, expected, tolerance);
        }
    }
}

/// Every default wait path now carries a deadline; none of them may become
/// observable on the success path.
///
/// The bound itself is exercised where the deadline can be driven below a real
/// submission's duration (`infrastructure::device::tests`). What this case
/// holds is the other half of that contract, over the public surface: with the
/// production bound in force, `download`, `download_owned`,
/// `download_sub_buffer`, `copy_buffer`, and `synchronize` return exactly what
/// they returned unbounded. It fails if a deadline is ever mis-plumbed into a
/// path — a zero or near-zero bound turns each of these into a timeout rather
/// than a value.
pub(crate) fn bounded_default_waits_leave_the_success_path_unchanged() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let expected: Vec<u32> = (0..1024).collect();
    let source = device.upload(&expected).unwrap();

    let owned = device.download_owned(&source).unwrap();
    assert_eq!(owned, expected);

    let mut into = vec![0_u32; expected.len()];
    device.download(&source, &mut into).unwrap();
    assert_eq!(into, expected);

    let mut tail = vec![0_u32; 16];
    device
        .download_sub_buffer(&source, 1000, &mut tail)
        .unwrap();
    assert_eq!(tail, expected[1000..1016]);

    let destination = device.alloc_uninitialized::<u32>(expected.len()).unwrap();
    device.copy_buffer(&source, &destination).unwrap();
    device.synchronize().unwrap();
    assert_eq!(device.download_owned(&destination).unwrap(), expected);

    // The explicit opt-out still routes to the same values when its own bound
    // is generous, so callers that need a different deadline are not on a
    // second code path.
    let mut explicit = vec![0_u32; expected.len()];
    device
        .download_with_timeout(&source, &mut explicit, std::time::Duration::from_secs(30))
        .unwrap();
    assert_eq!(explicit, expected);
}
