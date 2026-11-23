use super::*;
use hephaestus_core::{FftDirection, plan_fft_shape};

#[test]
fn capacity_is_exact_at_byte_and_workgroup_boundaries() {
    crate::test_support::run_cases(&[("exact_limits", exact_limits as fn())]);
}

fn limits(max_buffer_size: u64) -> DeviceLimits {
    DeviceLimits {
        max_buffer_size,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_storage_size: FUSED_WORKGROUP_STORAGE_BYTES,
        max_storage_buffers_per_shader_stage: Some(8),
        max_buffers_and_acceleration_structures_per_shader_stage: Some(8),
        max_immediate_size: 0,
    }
}

fn refusal<const R: usize>(
    plan: &FftPlan<R>,
    bytes: u64,
    binding_bytes: u64,
    groups: u32,
) -> (FftResource, u64, FftLimit, u64) {
    match Sizing::new::<f32, R>(plan, limits(bytes), binding_bytes, groups) {
        Err(HephaestusError::FftCapacityExceeded {
            resource,
            required,
            limiting_quantity,
            limit,
        }) => (resource, required, limiting_quantity, limit),
        _ => panic!("expected typed FFT capacity refusal"),
    }
}

fn exact_limits() {
    let plan =
        plan_fft_shape([17, 16, 16], FftDirection::Forward, &[0, 1, 2]).expect("dense FFT shape");
    let accepted = Sizing::new::<f32, 3>(&plan, limits(65_536), 65_536, 256)
        .expect("each storage and dispatch requirement fits its inclusive limit");
    assert_eq!(
        accepted.strategy,
        [
            AxisStrategy::ChirpZ { n: 17, m: 64 },
            AxisStrategy::FusedRadix2,
            AxisStrategy::FusedRadix2
        ]
    );
    assert_eq!(accepted.requirements.operand_bytes, 17 * 16 * 16 * 4);
    assert_eq!(
        accepted.requirements.workspace_bytes,
        Some(64 * 16 * 16 * 4)
    );
    assert_eq!(
        accepted.requirements.active_workgroups,
        [[64, 1], [256, 2], [256, 2]]
    );
    assert_eq!(
        accepted.requirements.chirp_kernel_bytes,
        [Some(64 * 4), None, None]
    );
    assert_eq!(
        accepted.requirements.chirp_factor_bytes,
        [Some(18 * 4), None, None]
    );
    assert_eq!(
        accepted.requirements.radix_twiddle_bytes,
        Some((64 + 7) * 4)
    );

    assert_eq!(
        refusal(&plan, 65_535, 65_536, 256),
        (
            FftResource::Workspace,
            65_536,
            FftLimit::BufferBytes,
            65_535
        )
    );
    assert_eq!(
        refusal(&plan, 65_536, 65_535, 256),
        (
            FftResource::Workspace,
            65_536,
            FftLimit::StorageBindingBytes,
            65_535
        )
    );

    let selected = plan_fft_shape([17, 16, 16], FftDirection::Inverse, &[0]).expect("single axis");
    assert!(Sizing::new::<f32, 3>(&selected, limits(65_536), u64::MAX, 64).is_ok());
    assert_eq!(
        refusal(&selected, 65_536, u64::MAX, 63),
        (
            FftResource::Dispatch,
            64,
            FftLimit::WorkgroupsPerDimension,
            63
        )
    );

    let large = plan_fft_shape([200, 200, 200], FftDirection::Forward, &[0, 1, 2])
        .expect("large FFT shape");
    assert_eq!(
        refusal(&large, u64::MAX, u64::MAX, 65_535),
        (
            FftResource::Dispatch,
            80_000,
            FftLimit::WorkgroupsPerDimension,
            65_535
        )
    );
}
