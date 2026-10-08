use eunomia::Pod;

use super::super::dialect::KernelDialect;
use super::parameters::{
    AdaGradParameters, AdaGradParametersF64, AdamParameters, AdamParametersF64, AdamWParameters,
    AdamWParametersF64, RmsPropParameters, RmsPropParametersF64, SgdParameters, SgdParametersF64,
};

mod sealed {
    pub trait Sealed {}
}

/// One value-level update step's outputs (ADR 0061 Decision 1): the next
/// parameter and the next value of each writable persistent-state slot.
///
/// Only `states[..Rule::STATE_COUNT]` is meaningful; a one-state rule's
/// unused second slot carries the unchanged input value, matching how
/// [`StatefulUpdateRule::BODY`]'s conditionally-rendered `state_one`
/// statements simply do not exist for that rule.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StatefulUpdateStep {
    /// `parameter_next`.
    pub parameter: f32,
    /// `state_zero_next` and, for a two-state rule, `state_one_next`.
    pub states: [f32; 2],
}

/// f64 twin of [`StatefulUpdateStep`]: one value-level update step's outputs
/// in double precision, with the same slot convention.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StatefulUpdateStepF64 {
    /// `parameter_next`.
    pub parameter: f64,
    /// `state_zero_next` and, for a two-state rule, `state_one_next`.
    pub states: [f64; 2],
}

/// Compile-time rule contract shared by all accelerator dialects.
pub trait StatefulUpdateRule<L: KernelDialect>:
    sealed::Sealed + Copy + Send + Sync + 'static
{
    /// Validated POD parameters uploaded once per dispatch.
    type Parameters: Pod;
    /// f64 twin of [`Self::Parameters`]: padding-free, same field order.
    type ParametersF64: Pod;
    /// Number of writable persistent-state views required by the rule.
    const STATE_COUNT: usize;
    /// Parameter field names in their packed host order, including padding.
    const PARAMETER_FIELDS: &'static [&'static str];
    /// f64 field names in packed order (no padding fields exist).
    const PARAMETER_FIELDS_F64: &'static [&'static str];
    /// Dialect-neutral scalar statements computing `parameter_next` and states.
    const BODY: &'static str;
    /// f64 body spelling, when [`Self::BODY`] pins `f32` literals (`1.0f`).
    /// `None` means the body is scalar-neutral (operand-only arithmetic) and
    /// backends reuse [`Self::BODY`] verbatim for f64 launches.
    const BODY_F64: Option<&'static str> = None;
    /// Revalidate a possibly byte-constructed parameter block before launch.
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()>;
    /// Revalidate a possibly byte-constructed f64 parameter block.
    fn validate_parameters_f64(parameters: &Self::ParametersF64)
    -> super::super::error::Result<()>;

    /// Apply one update step as a value function, the rule's definition
    /// (ADR 0061 Decision 1): exactly the arithmetic [`Self::BODY`] renders,
    /// in the same operation order. `StatefulUpdateRule` is sealed, so this
    /// is a required method directly on the trait rather than a
    /// `None`-defaulting seam the way the open operator families
    /// (`UnaryExpr`, `CombineExpr`, ...) work — every implementor is known,
    /// and the update is dialect-neutral scalar arithmetic, so the same
    /// `impl<L: KernelDialect> StatefulUpdateRule<L>` block below already
    /// covers [`Host`](crate::Host) with no separate blanket impl.
    fn update(
        parameter: f32,
        gradient: f32,
        states: [f32; 2],
        parameters: &Self::Parameters,
    ) -> StatefulUpdateStep;

    /// f64 twin of [`Self::update`]: exactly the arithmetic the f64 body
    /// renders, in the same operation order.
    fn update_f64(
        parameter: f64,
        gradient: f64,
        states: [f64; 2],
        parameters: &Self::ParametersF64,
    ) -> StatefulUpdateStepF64;
}

/// Stochastic gradient descent with momentum.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sgd;
/// Adam adaptive moment update.
#[derive(Clone, Copy, Debug, Default)]
pub struct Adam;
/// Adam with decoupled weight decay.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdamW;
/// RMSProp squared-gradient update.
#[derive(Clone, Copy, Debug, Default)]
pub struct RmsProp;
/// AdaGrad accumulated-squared-gradient update.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdaGrad;

impl sealed::Sealed for Sgd {}
impl sealed::Sealed for Adam {}
impl sealed::Sealed for AdamW {}
impl sealed::Sealed for RmsProp {}
impl sealed::Sealed for AdaGrad {}

impl<L: KernelDialect> StatefulUpdateRule<L> for Sgd {
    type Parameters = SgdParameters;
    type ParametersF64 = SgdParametersF64;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "momentum", "padding_zero", "padding_one"];
    const PARAMETER_FIELDS_F64: &'static [&'static str] = &["learning_rate", "momentum"];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.momentum + gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * state_zero_next;";
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
        parameters.validate()
    }
    fn validate_parameters_f64(
        parameters: &Self::ParametersF64,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update(
        parameter: f32,
        gradient: f32,
        states: [f32; 2],
        parameters: &Self::Parameters,
    ) -> StatefulUpdateStep {
        let state_zero_next = states[0] * parameters.momentum + gradient;
        let parameter_next = parameter - parameters.learning_rate * state_zero_next;
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }

    fn update_f64(
        parameter: f64,
        gradient: f64,
        states: [f64; 2],
        parameters: &Self::ParametersF64,
    ) -> StatefulUpdateStepF64 {
        let state_zero_next = states[0] * parameters.momentum + gradient;
        let parameter_next = parameter - parameters.learning_rate * state_zero_next;
        StatefulUpdateStepF64 {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for Adam {
    type Parameters = AdamParameters;
    type ParametersF64 = AdamParametersF64;
    const STATE_COUNT: usize = 2;
    const PARAMETER_FIELDS: &'static [&'static str] = &[
        "learning_rate",
        "beta_one",
        "beta_two",
        "epsilon",
        "bias_correction_one",
        "bias_correction_two",
        "padding_zero",
        "padding_one",
    ];
    const PARAMETER_FIELDS_F64: &'static [&'static str] = &[
        "learning_rate",
        "beta_one",
        "beta_two",
        "epsilon",
        "bias_correction_one",
        "bias_correction_two",
    ];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.beta_one + (1.0f - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + (1.0f - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);";
    const BODY_F64: Option<&'static str> = Some(
        "state_zero_next = state_zero_value * parameters.beta_one + (1.0 - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + (1.0 - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);",
    );
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
        parameters.validate()
    }
    fn validate_parameters_f64(
        parameters: &Self::ParametersF64,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update(
        parameter: f32,
        gradient: f32,
        states: [f32; 2],
        parameters: &Self::Parameters,
    ) -> StatefulUpdateStep {
        let state_zero_next =
            states[0] * parameters.beta_one + (1.0 - parameters.beta_one) * gradient;
        let state_one_next =
            states[1] * parameters.beta_two + (1.0 - parameters.beta_two) * gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one)
                / ((state_one_next / parameters.bias_correction_two).sqrt() + parameters.epsilon);
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, state_one_next],
        }
    }

    fn update_f64(
        parameter: f64,
        gradient: f64,
        states: [f64; 2],
        parameters: &Self::ParametersF64,
    ) -> StatefulUpdateStepF64 {
        let state_zero_next =
            states[0] * parameters.beta_one + (1.0 - parameters.beta_one) * gradient;
        let state_one_next =
            states[1] * parameters.beta_two + (1.0 - parameters.beta_two) * gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one)
                / ((state_one_next / parameters.bias_correction_two).sqrt() + parameters.epsilon);
        StatefulUpdateStepF64 {
            parameter: parameter_next,
            states: [state_zero_next, state_one_next],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for AdamW {
    type Parameters = AdamWParameters;
    type ParametersF64 = AdamWParametersF64;
    const STATE_COUNT: usize = 2;
    const PARAMETER_FIELDS: &'static [&'static str] = &[
        "learning_rate",
        "beta_one",
        "beta_two",
        "epsilon",
        "bias_correction_one",
        "bias_correction_two",
        "weight_decay",
        "padding",
    ];
    const PARAMETER_FIELDS_F64: &'static [&'static str] = &[
        "learning_rate",
        "beta_one",
        "beta_two",
        "epsilon",
        "bias_correction_one",
        "bias_correction_two",
        "weight_decay",
    ];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.beta_one + (1.0f - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + (1.0f - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value * (1.0f - parameters.learning_rate * parameters.weight_decay) - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);";
    const BODY_F64: Option<&'static str> = Some(
        "state_zero_next = state_zero_value * parameters.beta_one + (1.0 - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + (1.0 - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value * (1.0 - parameters.learning_rate * parameters.weight_decay) - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);",
    );
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
        parameters.validate()
    }
    fn validate_parameters_f64(
        parameters: &Self::ParametersF64,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update(
        parameter: f32,
        gradient: f32,
        states: [f32; 2],
        parameters: &Self::Parameters,
    ) -> StatefulUpdateStep {
        let state_zero_next =
            states[0] * parameters.beta_one + (1.0 - parameters.beta_one) * gradient;
        let state_one_next =
            states[1] * parameters.beta_two + (1.0 - parameters.beta_two) * gradient * gradient;
        let parameter_next = parameter * (1.0 - parameters.learning_rate * parameters.weight_decay)
            - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one)
                / ((state_one_next / parameters.bias_correction_two).sqrt() + parameters.epsilon);
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, state_one_next],
        }
    }

    fn update_f64(
        parameter: f64,
        gradient: f64,
        states: [f64; 2],
        parameters: &Self::ParametersF64,
    ) -> StatefulUpdateStepF64 {
        let state_zero_next =
            states[0] * parameters.beta_one + (1.0 - parameters.beta_one) * gradient;
        let state_one_next =
            states[1] * parameters.beta_two + (1.0 - parameters.beta_two) * gradient * gradient;
        let parameter_next = parameter * (1.0 - parameters.learning_rate * parameters.weight_decay)
            - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one)
                / ((state_one_next / parameters.bias_correction_two).sqrt() + parameters.epsilon);
        StatefulUpdateStepF64 {
            parameter: parameter_next,
            states: [state_zero_next, state_one_next],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for RmsProp {
    type Parameters = RmsPropParameters;
    type ParametersF64 = RmsPropParametersF64;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "alpha", "epsilon", "padding"];
    const PARAMETER_FIELDS_F64: &'static [&'static str] = &["learning_rate", "alpha", "epsilon"];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.alpha + (1.0f - parameters.alpha) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);";
    const BODY_F64: Option<&'static str> = Some(
        "state_zero_next = state_zero_value * parameters.alpha + (1.0 - parameters.alpha) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);",
    );
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
        parameters.validate()
    }
    fn validate_parameters_f64(
        parameters: &Self::ParametersF64,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update(
        parameter: f32,
        gradient: f32,
        states: [f32; 2],
        parameters: &Self::Parameters,
    ) -> StatefulUpdateStep {
        let state_zero_next =
            states[0] * parameters.alpha + (1.0 - parameters.alpha) * gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate * gradient / (state_zero_next.sqrt() + parameters.epsilon);
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }

    fn update_f64(
        parameter: f64,
        gradient: f64,
        states: [f64; 2],
        parameters: &Self::ParametersF64,
    ) -> StatefulUpdateStepF64 {
        let state_zero_next =
            states[0] * parameters.alpha + (1.0 - parameters.alpha) * gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate * gradient / (state_zero_next.sqrt() + parameters.epsilon);
        StatefulUpdateStepF64 {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for AdaGrad {
    type Parameters = AdaGradParameters;
    type ParametersF64 = AdaGradParametersF64;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "epsilon", "padding_zero", "padding_one"];
    const PARAMETER_FIELDS_F64: &'static [&'static str] = &["learning_rate", "epsilon"];
    const BODY: &'static str = "state_zero_next = state_zero_value + gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);";
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
        parameters.validate()
    }
    fn validate_parameters_f64(
        parameters: &Self::ParametersF64,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update(
        parameter: f32,
        gradient: f32,
        states: [f32; 2],
        parameters: &Self::Parameters,
    ) -> StatefulUpdateStep {
        let state_zero_next = states[0] + gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate * gradient / (state_zero_next.sqrt() + parameters.epsilon);
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }

    fn update_f64(
        parameter: f64,
        gradient: f64,
        states: [f64; 2],
        parameters: &Self::ParametersF64,
    ) -> StatefulUpdateStepF64 {
        let state_zero_next = states[0] + gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate * gradient / (state_zero_next.sqrt() + parameters.epsilon);
        StatefulUpdateStepF64 {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }
}

#[cfg(test)]
mod tests {
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
        let expected_state_one =
            0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
        let expected_parameter = 1.0
            - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
                / ((expected_state_one / parameters.bias_correction_two).sqrt()
                    + parameters.epsilon);
        assert_eq!(step.states, [expected_state_zero, expected_state_one]);
        assert_eq!(step.parameter, expected_parameter);
    }

    #[test]
    fn adamw_update_applies_decoupled_weight_decay_before_the_adam_step() {
        let parameters =
            AdamWParameters::new(0.1, 0.9, 0.999, 1.0e-6, 0.01, 3).expect("valid AdamW parameters");
        let step = <AdamW as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.25], &parameters);
        let expected_state_zero = 0.5 * parameters.beta_one + (1.0 - parameters.beta_one) * 2.0;
        let expected_state_one =
            0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
        let expected_parameter = 1.0 * (1.0 - parameters.learning_rate * parameters.weight_decay)
            - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
                / ((expected_state_one / parameters.bias_correction_two).sqrt()
                    + parameters.epsilon);
        assert_eq!(step.states, [expected_state_zero, expected_state_one]);
        assert_eq!(step.parameter, expected_parameter);
    }

    #[test]
    fn rmsprop_update_matches_hand_derived_arithmetic() {
        let parameters =
            RmsPropParameters::new(0.05, 0.9, 1.0e-6).expect("valid RMSProp parameters");
        let step = <RmsProp as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
        let expected_state_zero: f32 =
            0.5 * parameters.alpha + (1.0 - parameters.alpha) * 2.0 * 2.0;
        let expected_parameter = 1.0
            - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
        assert_eq!(step.states[0], expected_state_zero);
        assert_eq!(step.parameter, expected_parameter);
    }

    #[test]
    fn adagrad_update_accumulates_without_decay() {
        let parameters = AdaGradParameters::new(0.05, 1.0e-6).expect("valid AdaGrad parameters");
        let step = <AdaGrad as StatefulUpdateRule<Host>>::update(1.0, 2.0, [0.5, 0.0], &parameters);
        let expected_state_zero: f32 = 0.5 + 2.0 * 2.0;
        let expected_parameter = 1.0
            - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
        assert_eq!(step.states[0], expected_state_zero);
        assert_eq!(step.parameter, expected_parameter);
    }

    // f64 twins: same transcription discipline as above, in double
    // precision. Binary-exact values where the arithmetic allows it,
    // independently transcribed expressions otherwise.

    #[test]
    fn sgd_update_f64_matches_hand_derived_arithmetic() {
        let parameters = SgdParametersF64::new(0.25, 0.5).expect("valid f64 SGD parameters");
        let step =
            <Sgd as StatefulUpdateRule<Host>>::update_f64(10.0, 2.0, [0.0, 0.0], &parameters);
        assert_eq!(step.states, [2.0, 0.0]);
        assert_eq!(step.parameter, 9.5);
    }

    #[test]
    fn adam_update_f64_matches_the_bias_corrected_second_moment_formula() {
        let parameters =
            AdamParametersF64::new(0.1, 0.9, 0.999, 1.0e-6, 3).expect("valid f64 Adam parameters");
        let step =
            <Adam as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.25], &parameters);
        let expected_state_zero = 0.5 * parameters.beta_one + (1.0 - parameters.beta_one) * 2.0;
        let expected_state_one =
            0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
        let expected_parameter = 1.0
            - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
                / ((expected_state_one / parameters.bias_correction_two).sqrt()
                    + parameters.epsilon);
        assert_eq!(step.states, [expected_state_zero, expected_state_one]);
        assert_eq!(step.parameter, expected_parameter);
    }

    #[test]
    fn adamw_update_f64_applies_decoupled_weight_decay() {
        let parameters = AdamWParametersF64::new(0.1, 0.9, 0.999, 1.0e-6, 0.01, 3)
            .expect("valid f64 AdamW parameters");
        let step =
            <AdamW as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.25], &parameters);
        let expected_state_zero = 0.5 * parameters.beta_one + (1.0 - parameters.beta_one) * 2.0;
        let expected_state_one =
            0.25 * parameters.beta_two + (1.0 - parameters.beta_two) * 2.0 * 2.0;
        let expected_parameter = 1.0 * (1.0 - parameters.learning_rate * parameters.weight_decay)
            - parameters.learning_rate * (expected_state_zero / parameters.bias_correction_one)
                / ((expected_state_one / parameters.bias_correction_two).sqrt()
                    + parameters.epsilon);
        assert_eq!(step.states, [expected_state_zero, expected_state_one]);
        assert_eq!(step.parameter, expected_parameter);
    }

    #[test]
    fn rmsprop_update_f64_matches_hand_derived_arithmetic() {
        let parameters =
            RmsPropParametersF64::new(0.05, 0.9, 1.0e-6).expect("valid f64 RMSProp parameters");
        let step =
            <RmsProp as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.0], &parameters);
        let expected_state_zero: f64 =
            0.5 * parameters.alpha + (1.0 - parameters.alpha) * 2.0 * 2.0;
        let expected_parameter = 1.0
            - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
        assert_eq!(step.states[0], expected_state_zero);
        assert_eq!(step.parameter, expected_parameter);
    }

    #[test]
    fn adagrad_update_f64_accumulates_without_decay() {
        let parameters =
            AdaGradParametersF64::new(0.05, 1.0e-6).expect("valid f64 AdaGrad parameters");
        let step =
            <AdaGrad as StatefulUpdateRule<Host>>::update_f64(1.0, 2.0, [0.5, 0.0], &parameters);
        let expected_state_zero: f64 = 0.5 + 2.0 * 2.0;
        let expected_parameter = 1.0
            - parameters.learning_rate * 2.0 / (expected_state_zero.sqrt() + parameters.epsilon);
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
}
