use eunomia::Pod;

use super::super::dialect::KernelDialect;
use super::parameters::{
    AdaGradParameters, AdamParameters, AdamWParameters, RmsPropParameters, SgdParameters,
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

/// Compile-time rule contract shared by all accelerator dialects.
pub trait StatefulUpdateRule<L: KernelDialect>:
    sealed::Sealed + Copy + Send + Sync + 'static
{
    /// Validated POD parameters uploaded once per dispatch.
    type Parameters: Pod;
    /// Number of writable persistent-state views required by the rule.
    const STATE_COUNT: usize;
    /// Parameter field names in their packed host order, including padding.
    const PARAMETER_FIELDS: &'static [&'static str];
    /// Dialect-neutral scalar statements computing `parameter_next` and states.
    const BODY: &'static str;
    /// Revalidate a possibly byte-constructed parameter block before launch.
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()>;

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
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "momentum", "padding_zero", "padding_one"];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.momentum + gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * state_zero_next;";
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
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
}

impl<L: KernelDialect> StatefulUpdateRule<L> for Adam {
    type Parameters = AdamParameters;
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
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.beta_one + (1.0f - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + (1.0f - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);";
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
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
}

impl<L: KernelDialect> StatefulUpdateRule<L> for AdamW {
    type Parameters = AdamWParameters;
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
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.beta_one + (1.0f - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + (1.0f - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value * (1.0f - parameters.learning_rate * parameters.weight_decay) - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);";
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
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
}

impl<L: KernelDialect> StatefulUpdateRule<L> for RmsProp {
    type Parameters = RmsPropParameters;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "alpha", "epsilon", "padding"];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.alpha + (1.0f - parameters.alpha) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);";
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
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
}

impl<L: KernelDialect> StatefulUpdateRule<L> for AdaGrad {
    type Parameters = AdaGradParameters;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "epsilon", "padding_zero", "padding_one"];
    const BODY: &'static str = "state_zero_next = state_zero_value + gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);";
    fn validate_parameters(parameters: &Self::Parameters) -> super::super::error::Result<()> {
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
}
