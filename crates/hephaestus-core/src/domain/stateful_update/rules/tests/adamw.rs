use super::super::super::parameters::{AdamWParameters, AdamWParametersF64};
use super::super::{AdamW, StatefulUpdateRule};
use crate::domain::dialect::Host;

#[test]
fn adamw_update_applies_decoupled_weight_decay_before_the_adam_step() {
    let parameters =
        AdamWParameters::new(0.1, 0.9, 0.999, 1.0e-6, 0.01, 3).expect("valid AdamW parameters");
    let step = <AdamW as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.25], &parameters);
    let expected_state_zero = 0.5 * parameters.beta_one + (1.0 - parameters.beta_one) * 2.0;
    let expected_state_one = 0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
    let expected_parameter = 1.0 * (1.0 - parameters.learning_rate * parameters.weight_decay)
        - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
            / ((expected_state_one / parameters.bias_correction_two).sqrt() + parameters.epsilon);
    assert_eq!(step.states, [expected_state_zero, expected_state_one]);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn adamw_update_f64_applies_decoupled_weight_decay() {
    let parameters = AdamWParametersF64::new(0.1, 0.9, 0.999, 1.0e-6, 0.01, 3)
        .expect("valid f64 AdamW parameters");
    let step = <AdamW as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.25], &parameters);
    let expected_state_zero = 0.5 * parameters.beta_one + (1.0 - parameters.beta_one) * 2.0;
    let expected_state_one = 0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
    let expected_parameter = 1.0 * (1.0 - parameters.learning_rate * parameters.weight_decay)
        - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
            / ((expected_state_one / parameters.bias_correction_two).sqrt() + parameters.epsilon);
    assert_eq!(step.states, [expected_state_zero, expected_state_one]);
    assert_eq!(step.parameter, expected_parameter);
}
