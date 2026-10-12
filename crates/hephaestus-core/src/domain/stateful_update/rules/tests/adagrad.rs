use super::super::super::parameters::{AdaGradParameters, AdaGradParametersF64};
use super::super::{AdaGrad, StatefulUpdateRule};
use crate::domain::dialect::Host;

#[test]
fn adagrad_update_accumulates_without_decay() {
    let parameters = AdaGradParameters::new(0.05, 1.0e-6).expect("valid AdaGrad parameters");
    let step = <AdaGrad as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f32 = 0.5 + 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn adagrad_update_f64_accumulates_without_decay() {
    let parameters = AdaGradParametersF64::new(0.05, 1.0e-6).expect("valid f64 AdaGrad parameters");
    let step = <AdaGrad as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f64 = 0.5 + 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}
