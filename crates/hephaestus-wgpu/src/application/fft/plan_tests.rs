use super::*;
use hephaestus_core::HephaestusError;

#[test]
fn module_cases_share_process_state() {
    crate::test_support::run_cases(&[
        (
            "host_chirp_preparation_fft_matches_impulse_spectrum",
            host_chirp_preparation_fft_matches_impulse_spectrum as fn(),
        ),
        (
            "host_preparation_allocation_failure_is_typed",
            host_preparation_allocation_failure_is_typed as fn(),
        ),
        (
            "chirp_phase_is_range_reduced_before_precision_narrowing",
            chirp_phase_is_range_reduced_before_precision_narrowing as fn(),
        ),
    ]);
}

fn host_chirp_preparation_fft_matches_impulse_spectrum() {
    let mut values = vec![[0.0, 0.0]; 8];
    values[0] = [1.0, 0.0];
    forward_radix_two(&mut values);
    for value in values {
        assert!((value[0] - 1.0).abs() <= f64::EPSILON);
        assert!(value[1].abs() <= f64::EPSILON);
    }
}

fn host_preparation_allocation_failure_is_typed() {
    match try_host_vector::<u8>(usize::MAX, "test staging") {
        Err(HephaestusError::AllocationFailed { message }) => {
            assert!(message.contains("FFT test staging host allocation"));
            assert!(message.contains(&usize::MAX.to_string()));
        }
        other => panic!("expected typed allocation failure, got {other:?}"),
    }
}

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
