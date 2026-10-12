use super::super::super::dialect::KernelDialect;
use super::super::super::error::Result;
use super::super::parameters::{AdamParameters, AdamParametersF64};
use super::rule::{StatefulUpdateRule, StatefulUpdateStep, StatefulUpdateStepF64};

/// Adam adaptive moment update.
#[derive(Clone, Copy, Debug, Default)]
pub struct Adam;

impl super::sealed::Sealed for Adam {}

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
