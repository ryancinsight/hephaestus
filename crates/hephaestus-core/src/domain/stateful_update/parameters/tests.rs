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

#[test]
fn rejects_zeroed_adam_parameter_block() {
    let parameters = AdamParameters::zeroed();
    assert!(<Adam as StatefulUpdateRule<Wgsl>>::validate_parameters(&parameters).is_err());
}

#[test]
fn adam_bias_correction_matches_closed_form() {
    let parameters = AdamParameters::new(0.01, 0.9, 0.999, 1.0e-8, 3)
        .expect("analytically valid Adam parameters");
    assert_eq!(parameters.bias_correction_one, 1.0 - 0.9_f32.powi(3));
    assert_eq!(parameters.bias_correction_two, 1.0 - 0.999_f32.powi(3));
}

#[test]
fn rejects_nonfinite_and_out_of_domain_parameters() {
    assert_invalid(
        SgdParameters::new(f32::NAN, 0.0),
        "SGD parameter learning_rate must be finite, got NaN",
    );
    assert_invalid(
        SgdParameters::new(0.1, 1.0),
        "SGD parameter momentum must lie in [0, 1), got 1",
    );
    assert_invalid(
        AdamParameters::new(-0.1, 0.9, 0.999, 1.0e-8, 1),
        "Adam parameter learning_rate must be nonnegative, got -0.1",
    );
    assert_invalid(
        RmsPropParameters::new(0.1, 1.0, 1.0e-8),
        "RMSProp alpha must lie in [0, 1) and epsilon must be positive",
    );
    assert_invalid(
        AdaGradParameters::new(0.1, 0.0),
        "AdaGrad epsilon must be positive",
    );
    assert_invalid(
        AdamWParameters::new(0.1, 0.9, 0.999, 1.0e-8, -0.1, 1),
        "AdamW parameter weight_decay must be nonnegative, got -0.1",
    );
}

#[test]
fn all_rules_accept_zero_learning_rate() {
    SgdParameters::new(0.0, 0.9).expect("zero-rate SGD parameters");
    AdamParameters::new(0.0, 0.9, 0.999, 1.0e-8, 1).expect("zero-rate Adam parameters");
    AdamWParameters::new(0.0, 0.9, 0.999, 1.0e-8, 0.01, 1).expect("zero-rate AdamW parameters");
    RmsPropParameters::new(0.0, 0.9, 1.0e-8).expect("zero-rate RMSProp parameters");
    AdaGradParameters::new(0.0, 1.0e-8).expect("zero-rate AdaGrad parameters");
}

#[test]
fn f64_bias_correction_matches_closed_form() {
    let parameters = AdamParametersF64::new(0.01, 0.9, 0.999, 1.0e-8, 3)
        .expect("analytically valid f64 Adam parameters");
    assert_eq!(parameters.bias_correction_one, 1.0 - 0.9_f64.powi(3));
    assert_eq!(parameters.bias_correction_two, 1.0 - 0.999_f64.powi(3));
}

#[test]
fn f64_rejects_nonfinite_and_out_of_domain_parameters() {
    assert_invalid(
        SgdParametersF64::new(f64::NAN, 0.0),
        "SGD parameter learning_rate must be finite, got NaN",
    );
    assert_invalid(
        SgdParametersF64::new(0.1, 1.0),
        "SGD parameter momentum must lie in [0, 1), got 1",
    );
    assert_invalid(
        AdamParametersF64::new(-0.1, 0.9, 0.999, 1.0e-8, 1),
        "Adam parameter learning_rate must be nonnegative, got -0.1",
    );
    assert_invalid(
        RmsPropParametersF64::new(0.1, 1.0, 1.0e-8),
        "RMSProp alpha must lie in [0, 1) and epsilon must be positive",
    );
    assert_invalid(
        AdaGradParametersF64::new(0.1, 0.0),
        "AdaGrad epsilon must be positive",
    );
    assert_invalid(
        AdamWParametersF64::new(0.1, 0.9, 0.999, 1.0e-8, -0.1, 1),
        "AdamW parameter weight_decay must be nonnegative, got -0.1",
    );
}

#[test]
fn f64_rules_accept_zero_learning_rate() {
    SgdParametersF64::new(0.0, 0.9).expect("zero-rate f64 SGD parameters");
    AdamParametersF64::new(0.0, 0.9, 0.999, 1.0e-8, 1).expect("zero-rate f64 Adam parameters");
    AdamWParametersF64::new(0.0, 0.9, 0.999, 1.0e-8, 0.01, 1)
        .expect("zero-rate f64 AdamW parameters");
    RmsPropParametersF64::new(0.0, 0.9, 1.0e-8).expect("zero-rate f64 RMSProp parameters");
    AdaGradParametersF64::new(0.0, 1.0e-8).expect("zero-rate f64 AdaGrad parameters");
}

#[test]
fn f64_rejects_zeroed_adam_parameter_block() {
    let parameters = AdamParametersF64::zeroed();
    assert!(<Adam as StatefulUpdateRule<Wgsl>>::validate_parameters_f64(&parameters).is_err());
}
