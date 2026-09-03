use eunomia::Pod;

use super::super::dialect::KernelDialect;
use super::parameters::{
    AdaGradParameters, AdamParameters, AdamWParameters, RmsPropParameters, SgdParameters,
};

mod sealed {
    pub trait Sealed {}
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
}
