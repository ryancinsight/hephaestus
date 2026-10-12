use super::super::super::dialect::KernelDialect;
use super::super::super::error::Result;
use super::super::parameters::{AdaGradParameters, AdaGradParametersF64};
use super::rule::{StatefulUpdateRule, StatefulUpdateStep, StatefulUpdateStepF64};

/// AdaGrad accumulated-squared-gradient update.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdaGrad;

impl super::sealed::Sealed for AdaGrad {}

impl<L: KernelDialect> StatefulUpdateRule<L> for AdaGrad {
    type Parameters = AdaGradParameters;
    type ParametersF64 = AdaGradParametersF64;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "epsilon", "padding_zero", "padding_one"];
    const PARAMETER_FIELDS_F64: &'static [&'static str] = &["learning_rate", "epsilon"];
    const BODY: &'static str = "state_zero_next = state_zero_value + gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);";
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
