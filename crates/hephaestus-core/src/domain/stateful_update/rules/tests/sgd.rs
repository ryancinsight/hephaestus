use super::super::super::parameters::{SgdParameters, SgdParametersF64};
use super::super::{Sgd, StatefulUpdateRule};
use crate::domain::dialect::Host;

#[test]
fn sgd_update_matches_hand_derived_arithmetic_at_zero_state() {
    // learning_rate = 0.25 and momentum = 0.5 are exact in binary, so
    // every intermediate stays exact: state_zero_next = 0*0.5 + 2.0 =
    // 2.0; parameter_next = 10.0 - 0.25*2.0 = 9.5.
    let parameters = SgdParameters::new(0.25, 0.5).expect("valid SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update(10.0, 2.0, [0.0, 0.0], &parameters);
    assert_eq!(step.states, [2.0, 0.0]);
    assert_eq!(step.parameter, 9.5);
}

#[test]
fn sgd_update_is_identity_at_zero_gradient_and_zero_state() {
    // Multiplying by 0.0 and subtracting 0.0 are exact regardless of
    // `learning_rate`/`momentum`'s own rounding, so this holds for any
    // valid parameters.
    let parameters = SgdParameters::new(0.1, 0.9).expect("valid SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update(5.0, 0.0, [0.0, 0.0], &parameters);
    assert_eq!(step.parameter, 5.0);
    assert_eq!(step.states[0], 0.0);
}

#[test]
fn sgd_update_f64_matches_hand_derived_arithmetic() {
    let parameters = SgdParametersF64::new(0.25, 0.5).expect("valid f64 SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update_f64(10.0, 2.0, [0.0, 0.0], &parameters);
    assert_eq!(step.states, [2.0, 0.0]);
    assert_eq!(step.parameter, 9.5);
}
