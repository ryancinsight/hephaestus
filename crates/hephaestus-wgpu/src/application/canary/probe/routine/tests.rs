use std::sync::OnceLock;

use hephaestus_core::{DeviceFeature, DevicePreference, HephaestusError};

use super::{RELATIVE_ERROR_BOUND, evaluate};
use crate::application::canary::{PrecisionGrade, probe_exp_routine};
use crate::infrastructure::device::WgpuDevice;

/// Relative error contract of the CPU `exp` truth: one binary64 spacing,
/// `2^-52 = 2u`.
const REFERENCE_BOUND: f64 = f64::EPSILON;

/// Spacing of the subnormal range, `2^-1074`.
const SUBNORMAL_SPACING: f64 = f64::from_bits(1);

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let acquired =
                WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
                    "hephaestus-exp-routine-test",
                    DevicePreference::HighPerformance,
                    &[DeviceFeature::ShaderF64],
                    WgpuDevice::downlevel_device_limits(),
                );
            match acquired {
                Ok(device) if device.supports_device_feature(DeviceFeature::ShaderF64) => {
                    Some(device)
                }
                Ok(_) => {
                    tracing::info!("device lacks ShaderF64; exp routine tests skipped");
                    None
                }
                Err(error @ HephaestusError::AdapterUnavailable { .. }) => {
                    assert!(
                        std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none(),
                        "WGPU adapter required, but acquisition failed: {error}"
                    );
                    tracing::info!("adapter unavailable; exp routine tests skipped");
                    None
                }
                Err(error) => panic!("exp routine tests require a working provider: {error}"),
            }
        })
        .clone()
}

/// Bound on `|gpu - cpu| / |cpu|` from `|gpu - t| <= g |t|` and
/// `|cpu - t| <= r |t|`: `|gpu - cpu| <= (g + r) |t| <= (g + r) |cpu| / (1 - r)`.
fn comparison_bound() -> f64 {
    (RELATIVE_ERROR_BOUND + REFERENCE_BOUND) / (1.0 - REFERENCE_BOUND)
}

/// Assert `got` matches the CPU result for `input` within the derived bound;
/// a subnormal result carries two further subnormal spacings of rounding
/// (half from the final product, at most one from the CPU reference).
#[track_caller]
fn assert_matches_cpu(input: f64, got: f64) {
    let expected = input.exp();
    if expected.is_nan() {
        assert!(got.is_nan(), "exp({input:e}) must be NaN, got {got:e}");
    } else if expected.is_infinite() || expected == 0.0 {
        assert_eq!(got.to_bits(), expected.to_bits(), "exp({input:e})");
    } else {
        let allowance = comparison_bound() * expected
            + if expected < f64::MIN_POSITIVE {
                2.0 * SUBNORMAL_SPACING
            } else {
                0.0
            };
        assert!(
            (got - expected).abs() <= allowance,
            "exp({input:e}) = {got:e}, CPU {expected:e}, allowance {allowance:e}"
        );
    }
}

#[test]
fn routine_matches_cpu_truth_over_the_representable_range() {
    let subscriber = tracing_subscriber::fmt().with_test_writer().finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let Some(device) = device_or_skip() else {
        return;
    };
    let arrange = tracing::info_span!("arrange").entered();
    // Uniform grid over [-745, 709.5] whose step is not a power of two, so
    // consecutive points land on varied reduced arguments, plus a dense grid
    // over [-1, 1] where the reduction index is 0 or +-1.
    let wide: Vec<f64> = (0..=6144_u32)
        .map(|index| -745.0 + f64::from(index) * (1454.5 / 6144.0))
        .collect();
    let near: Vec<f64> = (0..=2048_u32)
        .map(|index| -1.0 + f64::from(index) * (2.0 / 2048.0))
        .collect();
    let inputs: Vec<f64> = wide.into_iter().chain(near).collect();
    drop(arrange);
    let outputs = {
        let _act = tracing::info_span!("act").entered();
        evaluate(&device, &inputs).expect("the routine dispatches on a ShaderF64 device")
    };
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(outputs.len(), inputs.len());
    for (input, got) in inputs.into_iter().zip(outputs) {
        assert_matches_cpu(input, got);
    }
}

#[test]
fn routine_special_values_match_cpu_truth() {
    let subscriber = tracing_subscriber::fmt().with_test_writer().finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let Some(device) = device_or_skip() else {
        return;
    };
    let arrange = tracing::info_span!("arrange").entered();
    let inputs = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.0e-300,
        -1.0e-300,
        709.0,
        709.7,
        710.0,
        710.5,
        1.0e300,
        -708.5,
        -740.0,
        -745.0,
        -746.0,
        -746.5,
        -1000.0,
        -1.0e300,
    ];
    drop(arrange);
    let outputs = {
        let _act = tracing::info_span!("act").entered();
        evaluate(&device, &inputs).expect("the routine dispatches on a ShaderF64 device")
    };
    let _assert = tracing::info_span!("assert").entered();
    for (input, got) in inputs.into_iter().zip(outputs) {
        assert_matches_cpu(input, got);
    }
}

#[test]
fn probe_grades_the_routine_exact_at_double_width() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let report = probe_exp_routine(&device).expect("the routine probe dispatches");
    assert_eq!(report.samples, 8);
    assert_eq!(
        report.grade(),
        PrecisionGrade::Exact,
        "worst relative error {:e}",
        report.max_relative_error
    );
    assert!(
        report.max_relative_error <= comparison_bound(),
        "worst relative error {:e} exceeds the derived bound {:e}",
        report.max_relative_error,
        comparison_bound()
    );
}
