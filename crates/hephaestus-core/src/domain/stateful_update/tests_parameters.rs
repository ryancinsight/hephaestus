use eunomia::Zeroable;

use super::*;
use crate::{Adam, StatefulUpdateRule, Wgsl};

fn assert_invalid<T>(result: Result<T>, expected: &str) {
    match result {
        Err(HephaestusError::InvalidConfiguration { message }) => {
            assert_eq!(message, expected)
        }
        Err(error) => panic!("expected InvalidConfiguration({expected:?}), got {error}"),
        Ok(_) => panic!("expected InvalidConfiguration({expected:?}), got success"),
    }
}

fn rejects_zeroed_adam_block<T: StatefulScalar>() {
    let parameters = AdamParameters::<T>::zeroed();
    assert!(<Adam as StatefulUpdateRule<Wgsl>>::validate_parameters(&parameters).is_err());
}

#[test]
fn rejects_zeroed_adam_parameter_block_at_both_widths() {
    rejects_zeroed_adam_block::<f32>();
    rejects_zeroed_adam_block::<f64>();
}

fn bias_correction_matches<T: StatefulScalar>(
    beta_one: T,
    beta_two: T,
    expected_one: T,
    expected_two: T,
) {
    let parameters = AdamParameters::new(
        T::from_f64(0.01),
        beta_one,
        beta_two,
        T::from_f64(1.0e-8),
        3,
    )
    .expect("analytically valid Adam parameters");
    assert_eq!(parameters.bias_correction_one(), expected_one);
    assert_eq!(parameters.bias_correction_two(), expected_two);
}

#[test]
fn adam_bias_correction_matches_closed_form_at_both_widths() {
    bias_correction_matches(
        0.9_f32,
        0.999_f32,
        1.0 - 0.9_f32.powi(3),
        1.0 - 0.999_f32.powi(3),
    );
    bias_correction_matches(
        0.9_f64,
        0.999_f64,
        1.0 - 0.9_f64.powi(3),
        1.0 - 0.999_f64.powi(3),
    );
}

fn rejects_out_of_domain<T: StatefulScalar>(
    nan: T,
    one: T,
    negative_rate: T,
    negative_decay: T,
    big_alpha: T,
    zero: T,
) {
    assert_invalid(
        SgdParameters::new(nan, T::ZERO),
        "SGD parameter learning_rate must be finite, got NaN",
    );
    assert_invalid(
        SgdParameters::new(T::from_f64(0.1), one),
        "SGD parameter momentum must lie in [0, 1), got 1",
    );
    assert_invalid(
        AdamParameters::new(
            negative_rate,
            T::from_f64(0.9),
            T::from_f64(0.999),
            T::from_f64(1.0e-8),
            1,
        ),
        "Adam parameter learning_rate must be nonnegative, got -0.1",
    );
    assert_invalid(
        RmsPropParameters::new(T::from_f64(0.1), big_alpha, T::from_f64(1.0e-8)),
        "RMSProp alpha must lie in [0, 1) and epsilon must be positive",
    );
    assert_invalid(
        AdaGradParameters::new(T::from_f64(0.1), zero),
        "AdaGrad epsilon must be positive",
    );
    assert_invalid(
        AdamWParameters::new(
            T::from_f64(0.1),
            T::from_f64(0.9),
            T::from_f64(0.999),
            T::from_f64(1.0e-8),
            negative_decay,
            1,
        ),
        "AdamW parameter weight_decay must be nonnegative, got -0.1",
    );
}

#[test]
fn rejects_nonfinite_and_out_of_domain_parameters_at_both_widths() {
    rejects_out_of_domain(f32::NAN, 1.0_f32, -0.1_f32, -0.1_f32, 1.0_f32, 0.0_f32);
    rejects_out_of_domain(f64::NAN, 1.0_f64, -0.1_f64, -0.1_f64, 1.0_f64, 0.0_f64);
}

fn accepts_zero_learning_rate<T: StatefulScalar>() {
    SgdParameters::new(T::ZERO, T::from_f64(0.9)).expect("zero-rate SGD parameters");
    AdamParameters::new(
        T::ZERO,
        T::from_f64(0.9),
        T::from_f64(0.999),
        T::from_f64(1.0e-8),
        1,
    )
    .expect("zero-rate Adam parameters");
    AdamWParameters::new(
        T::ZERO,
        T::from_f64(0.9),
        T::from_f64(0.999),
        T::from_f64(1.0e-8),
        T::from_f64(0.01),
        1,
    )
    .expect("zero-rate AdamW parameters");
    RmsPropParameters::new(T::ZERO, T::from_f64(0.9), T::from_f64(1.0e-8))
        .expect("zero-rate RMSProp parameters");
    AdaGradParameters::new(T::ZERO, T::from_f64(1.0e-8)).expect("zero-rate AdaGrad parameters");
}

#[test]
fn all_rules_accept_zero_learning_rate_at_both_widths() {
    accepts_zero_learning_rate::<f32>();
    accepts_zero_learning_rate::<f64>();
}
