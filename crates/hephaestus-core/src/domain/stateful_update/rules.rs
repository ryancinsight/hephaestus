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

    /// Body spelling for launches over scalar `T`: the f64 spelling where
    /// the f32 body pins `f32` literals, else [`Self::BODY`] verbatim.
    /// Backends call this with `f32`/`f64` only; any other scalar resolves
    /// to the f32 spelling, matching the f32-only [`Self::BODY`] contract.
    #[must_use]
    fn body_for<T: 'static>() -> &'static str {
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
            Self::BODY_F64.unwrap_or(Self::BODY)
        } else {
            Self::BODY
        }
    }

    /// Parameter field names for launches over scalar `T`: the padding-free
    /// list for f64, the padded uniform list otherwise (see
    /// [`Self::body_for`] for the width contract).
    #[must_use]
    fn parameter_fields_for<T: 'static>() -> &'static [&'static str] {
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
            Self::PARAMETER_FIELDS_F64
        } else {
            Self::PARAMETER_FIELDS
        }
    }
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
#[path = "tests_rules.rs"]
mod tests;
