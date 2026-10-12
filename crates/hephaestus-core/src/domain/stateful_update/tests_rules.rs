use super::super::super::dialect::Host;
use super::*;

// `update` is generic over every `L: KernelDialect` (the same impl covers
// `Host` with no separate blanket), so these tests pin it through `Host`. Each
// property runs at every admitted width: one generic helper per property,
// instantiated for `f32` and `f64`.

// Every expected value below is either exact binary arithmetic (powers of two
// and zero, which IEEE-754 multiply/add/subtract never round) or the same
// expression `update` computes, transcribed independently in the same
// operation order (ADR 0061 Decision 1).

fn sgd_hand_derived<T: StatefulScalar>() {
    // learning_rate = 0.25 and momentum = 0.5 are exact in binary, so every
    // intermediate stays exact: state_zero_next = 0*0.5 + 2.0 = 2.0;
    // parameter_next = 10.0 - 0.25*2.0 = 9.5.
    let parameters =
        SgdParameters::new(T::from_f64(0.25), T::from_f64(0.5)).expect("valid SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update(
        T::from_f64(10.0),
        T::from_f64(2.0),
        [T::ZERO, T::ZERO],
        &parameters,
    );
    assert_eq!(step.states, [T::from_f64(2.0), T::ZERO]);
    assert_eq!(step.parameter, T::from_f64(9.5));
}

fn sgd_identity_at_zero_gradient<T: StatefulScalar>() {
    // Multiplying by 0.0 and subtracting 0.0 are exact regardless of
    // `learning_rate`/`momentum`'s own rounding.
    let parameters =
        SgdParameters::new(T::from_f64(0.1), T::from_f64(0.9)).expect("valid SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update(
        T::from_f64(5.0),
        T::ZERO,
        [T::ZERO, T::ZERO],
        &parameters,
    );
    assert_eq!(step.parameter, T::from_f64(5.0));
    assert_eq!(step.states[0], T::ZERO);
}

fn adam_resets_at_zero_beta<T: StatefulScalar>() {
    // beta_one = beta_two = 0 forgets all prior state in one step
    // (`bias_correction = 1 - 0^step = 1` exactly for `step >= 1`), and at
    // gradient = 0 both moment estimates collapse to exactly 0.0, leaving the
    // parameter unchanged.
    let parameters = AdamParameters::new(T::from_f64(0.1), T::ZERO, T::ZERO, T::ONE, 1)
        .expect("valid Adam parameters");
    let step = <Adam as StatefulUpdateRule<Host>>::update(
        T::from_f64(10.0),
        T::ZERO,
        [T::from_f64(3.0), T::from_f64(4.0)],
        &parameters,
    );
    assert_eq!(step.states, [T::ZERO, T::ZERO]);
    assert_eq!(step.parameter, T::from_f64(10.0));
}

fn adam_matches_bias_corrected_formula<T: StatefulScalar>() {
    let parameters = AdamParameters::new(
        T::from_f64(0.1),
        T::from_f64(0.9),
        T::from_f64(0.999),
        T::from_f64(1.0e-6),
        3,
    )
    .expect("valid Adam parameters");
    let two = T::from_f64(2.0);
    let step = <Adam as StatefulUpdateRule<Host>>::update(
        T::ONE,
        two,
        [T::from_f64(0.5), T::from_f64(0.25)],
        &parameters,
    );
    let expected_state_zero =
        T::from_f64(0.5) * parameters.beta_one() + (T::ONE - parameters.beta_one()) * two;
    let expected_state_one =
        T::from_f64(0.25) * parameters.beta_two() + (T::ONE - parameters.beta_two()) * two * two;
    let expected_parameter = T::ONE
        - parameters.learning_rate() * (expected_state_zero / parameters.bias_correction_one())
            / ((expected_state_one / parameters.bias_correction_two()).sqrt()
                + parameters.epsilon());
    assert_eq!(step.states, [expected_state_zero, expected_state_one]);
    assert_eq!(step.parameter, expected_parameter);
}

fn adamw_applies_decoupled_weight_decay<T: StatefulScalar>() {
    let parameters = AdamWParameters::new(
        T::from_f64(0.1),
        T::from_f64(0.9),
        T::from_f64(0.999),
        T::from_f64(1.0e-6),
        T::from_f64(0.01),
        3,
    )
    .expect("valid AdamW parameters");
    let two = T::from_f64(2.0);
    let step = <AdamW as StatefulUpdateRule<Host>>::update(
        T::ONE,
        two,
        [T::from_f64(0.5), T::from_f64(0.25)],
        &parameters,
    );
    let expected_state_zero =
        T::from_f64(0.5) * parameters.beta_one() + (T::ONE - parameters.beta_one()) * two;
    let expected_state_one =
        T::from_f64(0.25) * parameters.beta_two() + (T::ONE - parameters.beta_two()) * two * two;
    let expected_parameter = T::ONE
        * (T::ONE - parameters.learning_rate() * parameters.weight_decay())
        - parameters.learning_rate() * (expected_state_zero / parameters.bias_correction_one())
            / ((expected_state_one / parameters.bias_correction_two()).sqrt()
                + parameters.epsilon());
    assert_eq!(step.states, [expected_state_zero, expected_state_one]);
    assert_eq!(step.parameter, expected_parameter);
}

fn rmsprop_matches_hand_derived_arithmetic<T: StatefulScalar>() {
    let parameters =
        RmsPropParameters::new(T::from_f64(0.05), T::from_f64(0.9), T::from_f64(1.0e-6))
            .expect("valid RMSProp parameters");
    let two = T::from_f64(2.0);
    let step = <RmsProp as StatefulUpdateRule<Host>>::update(
        T::ONE,
        two,
        [T::from_f64(0.5), T::ZERO],
        &parameters,
    );
    let expected_state_zero =
        T::from_f64(0.5) * parameters.alpha() + (T::ONE - parameters.alpha()) * two * two;
    let expected_parameter = T::ONE
        - parameters.learning_rate() * two / (expected_state_zero.sqrt() + parameters.epsilon());
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

fn adagrad_accumulates_without_decay<T: StatefulScalar>() {
    let parameters = AdaGradParameters::new(T::from_f64(0.05), T::from_f64(1.0e-6))
        .expect("valid AdaGrad parameters");
    let two = T::from_f64(2.0);
    let step = <AdaGrad as StatefulUpdateRule<Host>>::update(
        T::ONE,
        two,
        [T::from_f64(0.5), T::ZERO],
        &parameters,
    );
    let expected_state_zero = T::from_f64(0.5) + two * two;
    let expected_parameter = T::ONE
        - parameters.learning_rate() * two / (expected_state_zero.sqrt() + parameters.epsilon());
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn sgd_update_matches_hand_derived_arithmetic_at_both_widths() {
    sgd_hand_derived::<f32>();
    sgd_hand_derived::<f64>();
}

#[test]
fn sgd_update_is_identity_at_zero_gradient_at_both_widths() {
    sgd_identity_at_zero_gradient::<f32>();
    sgd_identity_at_zero_gradient::<f64>();
}

#[test]
fn adam_update_resets_at_zero_beta_at_both_widths() {
    adam_resets_at_zero_beta::<f32>();
    adam_resets_at_zero_beta::<f64>();
}

#[test]
fn adam_update_matches_the_bias_corrected_second_moment_formula_at_both_widths() {
    adam_matches_bias_corrected_formula::<f32>();
    adam_matches_bias_corrected_formula::<f64>();
}

#[test]
fn adamw_update_applies_decoupled_weight_decay_at_both_widths() {
    adamw_applies_decoupled_weight_decay::<f32>();
    adamw_applies_decoupled_weight_decay::<f64>();
}

#[test]
fn rmsprop_update_matches_hand_derived_arithmetic_at_both_widths() {
    rmsprop_matches_hand_derived_arithmetic::<f32>();
    rmsprop_matches_hand_derived_arithmetic::<f64>();
}

#[test]
fn adagrad_update_accumulates_without_decay_at_both_widths() {
    adagrad_accumulates_without_decay::<f32>();
    adagrad_accumulates_without_decay::<f64>();
}

#[test]
fn body_renders_the_width_literal_for_every_rule() {
    // f32 keeps the `f`-suffixed unit literal; f64 spells it without suffix.
    // Both widths render the same statements, so only the literal differs.
    for body in [
        <Adam as StatefulUpdateRule<Host>>::body_for::<f32>(),
        <AdamW as StatefulUpdateRule<Host>>::body_for::<f32>(),
        <RmsProp as StatefulUpdateRule<Host>>::body_for::<f32>(),
    ] {
        assert!(body.contains("1.0f - parameters."));
        assert!(!body.contains("{one}"));
    }
    for body in [
        <Adam as StatefulUpdateRule<Host>>::body_for::<f64>(),
        <AdamW as StatefulUpdateRule<Host>>::body_for::<f64>(),
        <RmsProp as StatefulUpdateRule<Host>>::body_for::<f64>(),
    ] {
        assert!(body.contains("(1.0 - parameters."));
        assert!(!body.contains("1.0f"));
        assert!(!body.contains("{one}"));
    }
    // Operand-only bodies carry no unit literal at either width.
    assert!(!<Sgd as StatefulUpdateRule<Host>>::body_for::<f32>().contains("1.0"));
    assert!(!<AdaGrad as StatefulUpdateRule<Host>>::body_for::<f64>().contains("1.0"));
}

#[test]
fn parameter_fields_are_width_independent() {
    // One field list renders the uniform block at every width: the padding
    // slots are explicit at both, so the list needs no per-width variant.
    assert_eq!(
        <Sgd as StatefulUpdateRule<Host>>::PARAMETER_FIELDS,
        &["learning_rate", "momentum", "padding_zero", "padding_one"]
    );
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len(),
        8
    );
    assert_eq!(
        <AdamW as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len(),
        8
    );
    assert_eq!(
        <RmsProp as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len(),
        4
    );
    assert_eq!(
        <AdaGrad as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len(),
        4
    );
}
