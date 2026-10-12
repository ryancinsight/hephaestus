use super::super::super::dialect::KernelDialect;
use super::super::super::error::Result;
use super::super::parameters::{SgdParameters, SgdParametersF64};
use super::rule::{StatefulUpdateRule, StatefulUpdateStep, StatefulUpdateStepF64};

/// Stochastic gradient descent with momentum.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sgd;

impl super::sealed::Sealed for Sgd {}

impl<L: KernelDialect> StatefulUpdateRule<L> for Sgd {
    type Parameters = SgdParameters;
    type ParametersF64 = SgdParametersF64;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "momentum", "padding_zero", "padding_one"];
    const PARAMETER_FIELDS_F64: &'static [&'static str] = &["learning_rate", "momentum"];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.momentum + gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * state_zero_next;";
    fn validate_parameters(parameters: &Self::Parameters) -> Result<()> {
        parameters.validate()
    }
    fn validate_parameters_f64(parameters: &Self::ParametersF64) -> Result<()> {
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
