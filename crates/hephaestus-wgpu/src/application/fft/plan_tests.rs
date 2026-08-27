use super::*;

#[test]
fn strategy_selects_radix_and_bluestein_without_rank_clones() {
    assert_eq!(
        axis_strategy_for(64).expect("valid radix strategy"),
        AxisStrategy::StagedRadix2
    );
    assert_eq!(
        axis_strategy_for(12).expect("valid Bluestein strategy"),
        AxisStrategy::ChirpZ { n: 12, m: 32 }
    );
}

#[test]
fn host_chirp_preparation_fft_matches_impulse_spectrum() {
    let mut values = vec![[0.0, 0.0]; 8];
    values[0] = [1.0, 0.0];
    forward_radix_two(&mut values);
    for value in values {
        assert!((value[0] - 1.0).abs() <= f64::EPSILON);
        assert!(value[1].abs() <= f64::EPSILON);
    }
}

#[test]
fn workspace_accounts_for_axis_batch_geometry() {
    let dimensions = [2, 3, 4];
    assert_eq!(
        axis_workspace_elements(dimensions, Axis::Y, AxisStrategy::ChirpZ { n: 3, m: 8 })
            .expect("valid workspace geometry"),
        Some(64)
    );
}

#[test]
fn fused_radix_eliminates_workspace_when_device_limits_allow_it() {
    let limits = DeviceLimits {
        max_buffer_size: u64::MAX,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_storage_size: FUSED_WORKGROUP_STORAGE_BYTES,
        max_storage_buffers_per_shader_stage: Some(8),
        max_buffers_and_acceleration_structures_per_shader_stage: Some(8),
        max_immediate_size: 0,
    };
    let strategy = select_fused_strategy(AxisStrategy::StagedRadix2, 256, 16_384, limits, 65_535);
    assert_eq!(strategy, AxisStrategy::FusedRadix2);
    assert_eq!(
        validate_storage_limit(
            u64::MAX,
            65_535,
            256 * 128 * 128,
            [256, 128, 128],
            [AxisStrategy::FusedRadix2; 3],
        )
        .expect("fused geometry is valid"),
        None
    );
}

#[test]
fn storage_validation_covers_bluestein_workspace_and_dispatch_limits() {
    let dimensions = [3, 1, 1];
    let strategies = [
        AxisStrategy::ChirpZ { n: 3, m: 8 },
        AxisStrategy::StagedRadix2,
        AxisStrategy::StagedRadix2,
    ];
    assert!(validate_storage_limit(31, u32::MAX, 3, dimensions, strategies).is_err());
    assert!(validate_storage_limit(u64::MAX, 0, 3, dimensions, strategies).is_err());
    assert_eq!(
        validate_storage_limit(u64::MAX, 1, 3, dimensions, strategies)
            .expect("one workgroup covers the prepared workspace"),
        Some(8)
    );
    assert!(
        validate_storage_limit(
            u64::MAX,
            65_535,
            1 << 24,
            [1 << 24, 1, 1],
            [AxisStrategy::StagedRadix2; 3],
        )
        .is_err()
    );
}

#[test]
fn host_preparation_allocation_failure_is_typed() {
    match try_host_vector::<u8>(usize::MAX, "test staging") {
        Err(HephaestusError::AllocationFailed { message }) => {
            assert!(message.contains("FFT test staging host allocation"));
            assert!(message.contains(&usize::MAX.to_string()));
        }
        other => panic!("expected typed allocation failure, got {other:?}"),
    }
}

#[test]
fn chirp_phase_is_range_reduced_before_precision_narrowing() {
    let n = 1_000_003_u32;
    let index = 980_700_u32;
    let reduced = chirp_angle(index, n);
    let direct = core::f64::consts::PI * f64::from(index) * f64::from(index) / f64::from(n);
    // The unreduced reference's argument reduction grows with its roughly
    // three-million-radian input; bound that path by epsilon times angle.
    let reference_bound = 16.0 * f64::EPSILON * direct.abs().max(1.0);
    assert!((reduced.cos() - direct.cos()).abs() <= reference_bound);
    assert!((reduced.sin() - direct.sin()).abs() <= reference_bound);

    let shader_precision = core::f32::consts::PI * index as f32 * index as f32 / n as f32;
    let shader_reduced = f64::from(shader_precision).rem_euclid(core::f64::consts::TAU);
    let phase_delta = (shader_reduced - reduced).abs();
    let wrapped_delta = phase_delta.min(core::f64::consts::TAU - phase_delta);
    assert!(
        wrapped_delta > 0.05,
        "regression input must distinguish range-reduced preparation from shader f32 phase construction"
    );
}
