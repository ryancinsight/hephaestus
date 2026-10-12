use super::super::super::dialect::Host;
use super::*;

// `update` is generic over every `L: KernelDialect` (the same impl
// covers `Host` with no separate blanket, per this trait's doc comment),
// so these tests pin it through `Host` specifically — the dialect this
// ADR increment adds host dispatch for.

// Every expected value below is either exact binary arithmetic (powers
// of two and zero, which IEEE-754 multiply/add/subtract never round) or
// the same expression `update` computes, transcribed independently but
// reading the same constructed `Parameters` fields and literal operands
// (ADR 0061 Decision 1: "the exact update the WGSL body computes, in the
// same operation order") — so both sides evaluate to bit-identical f32
// results, and a regression in operand order or grouping changes the
// transcription along with the implementation only if both are edited
// identically, which a reviewer would notice.

#[test]
fn sgd_update_matches_hand_derived_arithmetic_at_zero_state() {
    // learning_rate = 0.25 and momentum = 0.5 are exact in binary, so
    // every intermediate stays exact: state_zero_next = 0*0.5 + 2.0 =
    // 2.0; parameter_next = 10.0 - 0.25*2.0 = 9.5.
    let parameters = SgdParameters::<f32>::new(0.25, 0.5).expect("valid SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update(10.0, 2.0, [0.0, 0.0], &parameters);
    assert_eq!(step.states, [2.0, 0.0]);
    assert_eq!(step.parameter, 9.5);
}

#[test]
fn sgd_update_is_identity_at_zero_gradient_and_zero_state() {
    // Multiplying by 0.0 and subtracting 0.0 are exact regardless of
    // `learning_rate`/`momentum`'s own rounding, so this holds for any
    // valid parameters.
    let parameters = SgdParameters::<f32>::new(0.1, 0.9).expect("valid SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update(5.0, 0.0, [0.0, 0.0], &parameters);
    assert_eq!(step.parameter, 5.0);
    assert_eq!(step.states[0], 0.0);
}

#[test]
fn adam_update_resets_at_zero_beta_and_zero_gradient() {
    // beta_one = beta_two = 0 forgets all prior state in one step
    // (`bias_correction = 1 - 0^step = 1` exactly for `step >= 1`), and
    // at gradient = 0 both moment estimates collapse to exactly 0.0
    // regardless of the prior state's magnitude, leaving the parameter
    // unchanged (0.0 numerator over a nonzero denominator).
    let parameters =
        AdamParameters::<f32>::new(0.1, 0.0, 0.0, 1.0, 1).expect("valid Adam parameters");
    let step = <Adam as StatefulUpdateRule<Host>>::update(10.0, 0.0, [3.0, 4.0], &parameters);
    assert_eq!(step.states, [0.0, 0.0]);
    assert_eq!(step.parameter, 10.0);
}

#[test]
fn adam_update_matches_the_bias_corrected_second_moment_formula() {
    let parameters =
        AdamParameters::<f32>::new(0.1, 0.9, 0.999, 1.0e-6, 3).expect("valid Adam parameters");
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
fn adamw_update_applies_decoupled_weight_decay_before_the_adam_step() {
    let parameters = AdamWParameters::<f32>::new(0.1, 0.9, 0.999, 1.0e-6, 0.01, 3)
        .expect("valid AdamW parameters");
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
fn rmsprop_update_matches_hand_derived_arithmetic() {
    let parameters =
        RmsPropParameters::<f32>::new(0.05, 0.9, 1.0e-6).expect("valid RMSProp parameters");
    let step = <RmsProp as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f32 = 0.5 * parameters.alpha + (1.0 - parameters.alpha) * 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn adagrad_update_accumulates_without_decay() {
    let parameters = AdaGradParameters::<f32>::new(0.05, 1.0e-6).expect("valid AdaGrad parameters");
    let step = <AdaGrad as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f32 = 0.5 + 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

// f64 twins: same transcription discipline as above, in double
// precision. Binary-exact values where the arithmetic allows it,
// independently transcribed expressions otherwise.

#[test]
fn sgd_update_f64_matches_hand_derived_arithmetic() {
    let parameters = SgdParameters::<f64>::new(0.25, 0.5).expect("valid f64 SGD parameters");
    let step = <Sgd as StatefulUpdateRule<Host>>::update(10.0, 2.0, [0.0, 0.0], &parameters);
    assert_eq!(step.states, [2.0, 0.0]);
    assert_eq!(step.parameter, 9.5);
}

#[test]
fn adam_update_f64_matches_the_bias_corrected_second_moment_formula() {
    let parameters =
        AdamParameters::<f64>::new(0.1, 0.9, 0.999, 1.0e-6, 3).expect("valid f64 Adam parameters");
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
fn adamw_update_f64_applies_decoupled_weight_decay() {
    let parameters = AdamWParameters::<f64>::new(0.1, 0.9, 0.999, 1.0e-6, 0.01, 3)
        .expect("valid f64 AdamW parameters");
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
fn rmsprop_update_f64_matches_hand_derived_arithmetic() {
    let parameters =
        RmsPropParameters::<f64>::new(0.05, 0.9, 1.0e-6).expect("valid f64 RMSProp parameters");
    let step = <RmsProp as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f64 = 0.5 * parameters.alpha + (1.0 - parameters.alpha) * 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn adagrad_update_f64_accumulates_without_decay() {
    let parameters =
        AdaGradParameters::<f64>::new(0.05, 1.0e-6).expect("valid f64 AdaGrad parameters");
    let step = <AdaGrad as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
    let expected_state_zero: f64 = 0.5 + 2.0 * 2.0;
    let expected_parameter =
        1.0 - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
    assert_eq!(step.states[0], expected_state_zero);
    assert_eq!(step.parameter, expected_parameter);
}

#[test]
fn f64_bodies_exist_exactly_where_f32_literals_appear() {
    // Adam-family bodies pin `1.0f`; their f64 spellings use `1.0`
    // (AbstractFloat in WGSL, double in kernel C) and carry no `f`
    // suffix anywhere.
    for body in [
        <Adam as StatefulUpdateRule<Host>>::BODY_F64,
        <AdamW as StatefulUpdateRule<Host>>::BODY_F64,
        <RmsProp as StatefulUpdateRule<Host>>::BODY_F64,
    ] {
        let body = body.expect("f32-literal rule needs a BODY_F64");
        assert!(body.contains("1.0"), "f64 body must spell unit literals");
        assert!(
            !body.contains("1.0f"),
            "f64 body must not carry f32-suffixed literals"
        );
    }
    // SGD and AdaGrad bodies are operand-only arithmetic; backends
    // reuse BODY verbatim.
    assert!(<Sgd as StatefulUpdateRule<Host>>::BODY_F64.is_none());
    assert!(<AdaGrad as StatefulUpdateRule<Host>>::BODY_F64.is_none());
    assert!(!<Sgd as StatefulUpdateRule<Host>>::BODY.contains("1.0f"));
    assert!(!<AdaGrad as StatefulUpdateRule<Host>>::BODY.contains("1.0f"));
}

#[test]
fn width_selectors_resolve_both_spellings() {
    // f32 resolves to the f32-only items for every rule.
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::body_for::<f32>(),
        <Adam as StatefulUpdateRule<Host>>::BODY
    );
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::parameter_fields_for::<f32>(),
        <Adam as StatefulUpdateRule<Host>>::PARAMETER_FIELDS
    );
    // f64 resolves to the f64 spelling where one exists, else BODY.
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::body_for::<f64>(),
        <Adam as StatefulUpdateRule<Host>>::BODY_F64.expect("Adam ships BODY_F64")
    );
    assert_eq!(
        <Sgd as StatefulUpdateRule<Host>>::body_for::<f64>(),
        <Sgd as StatefulUpdateRule<Host>>::BODY
    );
    assert_eq!(
        <Sgd as StatefulUpdateRule<Host>>::parameter_fields_for::<f64>(),
        <Sgd as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64
    );
}

#[test]
fn f64_field_lists_drop_only_the_padding() {
    assert_eq!(
        <Sgd as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64,
        &["learning_rate", "momentum"]
    );
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <Adam as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 2
    );
    assert_eq!(
        <AdamW as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <AdamW as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 1
    );
    assert_eq!(
        <RmsProp as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <RmsProp as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 1
    );
    assert_eq!(
        <AdaGrad as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <AdaGrad as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 2
    );
}
