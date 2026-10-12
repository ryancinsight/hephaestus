use super::super::super::parameters::{AdamParameters, AdamParametersF64};
use super::super::{Adam, StatefulUpdateRule};
use crate::domain::dialect::Host;

#[test]
fn adam_update_resets_at_zero_beta_and_zero_gradient() {
    // beta_one = beta_two = 0 forgets all prior state in one step
    // (`bias_correction = 1 - 0^step = 1` exactly for `step >= 1`), and
    // at gradient = 0 both moment estimates collapse to exactly 0.0
    // regardless of the prior state's magnitude, leaving the parameter
    // unchanged (0.0 numerator over a nonzero denominator).
    let parameters = AdamParameters::new(0.1, 0.0, 0.0, 1.0, 1).expect("valid Adam parameters");
    let step = <Adam as StatefulUpdateRule<Host>>::update(10.0, 0.0, [3.0, 4.0], &parameters);
    assert_eq!(step.states, [0.0, 0.0]);
    assert_eq!(step.parameter, 10.0);
}

#[test]
fn adam_update_matches_the_bias_corrected_second_moment_formula() {
    let parameters =
        AdamParameters::new(0.1, 0.9, 0.999, 1.0e-6, 3).expect("valid Adam parameters");
    let step = <Adam as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.25], &parameters);
    let expected_state_zero = 0.5 * parameters.beta_one + (1.0 - parameters.beta_one) * 2.0;
    let expected_state_one = 0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
    let expected_parameter = 1.0
        - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
            / ((expected_state_one / parameters.bias_correction_two).sqrt() + parameters.epsilon);
    assert_eq!(step.states, [expected_state_zero, expected_state_one]);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn adam_update_f64_matches_the_bias_corrected_second_moment_formula() {
    let parameters =
        AdamParametersF64::new(0.1, 0.9, 0.999, 1.0e-6, 3).expect("valid f64 Adam parameters");
    let step = <Adam as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.25], &parameters);
    let expected_state_zero = 0.5 * parameters.beta_one + (1.0 - parameters.beta_one) * 2.0;
    let expected_state_one = 0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
    let expected_parameter = 1.0
        - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
            / ((expected_state_one / parameters.bias_correction_two).sqrt() + parameters.epsilon);
    assert_eq!(step.states, [expected_state_zero, expected_state_one]);
    assert_eq!(step.parameter, expected_parameter);
}
