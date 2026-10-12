use super::super::super::parameters::{RmsPropParameters, RmsPropParametersF64};
use super::super::{RmsProp, StatefulUpdateRule};
use crate::domain::dialect::Host;

#[test]
fn rmsprop_update_matches_hand_derived_arithmetic() {
    let parameters = RmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("valid RMSProp parameters");
    let step = <RmsProp as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f32 = 0.5 * parameters.alpha + (1.0 - parameters.alpha) * 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn rmsprop_update_f64_matches_hand_derived_arithmetic() {
    let parameters =
        RmsPropParametersF64::new(0.05, 0.9, 1.0e-6).expect("valid f64 RMSProp parameters");
    let step = <RmsProp as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f64 = 0.5 * parameters.alpha + (1.0 - parameters.alpha) * 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}
