use super::super::super::dialect::KernelDialect;
use super::super::super::error::Result;
use super::super::parameters::{RmsPropParameters, RmsPropParametersF64};
use super::rule::{StatefulUpdateRule, StatefulUpdateStep, StatefulUpdateStepF64};

/// RMSProp squared-gradient update.
#[derive(Clone, Copy, Debug, Default)]
pub struct RmsProp;

impl super::sealed::Sealed for RmsProp {}

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
