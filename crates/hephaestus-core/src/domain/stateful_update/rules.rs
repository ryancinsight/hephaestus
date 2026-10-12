use eunomia::Pod;

use super::super::dialect::KernelDialect;
use super::parameters::{
    AdaGradParameters, AdamParameters, AdamWParameters, RmsPropParameters, SgdParameters,
};
use super::scalar::StatefulScalar;

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
pub struct StatefulUpdateStep<T> {
    /// `parameter_next`.
    pub parameter: T,
    /// `state_zero_next` and, for a two-state rule, `state_one_next`.
    pub states: [T; 2],
}

/// Compile-time rule contract shared by all accelerator dialects.
pub trait StatefulUpdateRule<L: KernelDialect>:
    sealed::Sealed + Copy + Send + Sync + 'static
{
    /// Validated POD parameters for scalar `T`, uploaded once per dispatch.
    type Parameters<T: StatefulScalar>: Pod;
    /// Number of writable persistent-state views required by the rule.
    const STATE_COUNT: usize;
    /// Parameter field names in packed order, including padding fields. The
    /// same list renders the uniform block at every admitted width.
    const PARAMETER_FIELDS: &'static [&'static str];
    /// Dialect-neutral scalar statements computing `parameter_next` and states.
    ///
    /// The literal `{one}` marks `1.0`; [`Self::body_for`] substitutes the
    /// width's [`StatefulScalar::ONE_LITERAL`] when a kernel is rendered.
    const BODY: &'static str;

    /// Revalidate a possibly byte-constructed parameter block before launch.
    fn validate_parameters<T: StatefulScalar>(
        parameters: &Self::Parameters<T>,
    ) -> super::super::error::Result<()>;

    /// Apply one update step as a value function, the rule's definition
    /// (ADR 0061 Decision 1): exactly the arithmetic [`Self::BODY`] renders,
    /// in the same operation order. `StatefulUpdateRule` is sealed, so this
    /// is a required method directly on the trait rather than a
    /// `None`-defaulting seam the way the open operator families
    /// (`UnaryExpr`, `CombineExpr`, ...) work — every implementor is known,
    /// and the update is dialect-neutral scalar arithmetic, so the same
    /// `impl<L: KernelDialect> StatefulUpdateRule<L>` block below already
    /// covers [`Host`](crate::Host) with no separate blanket impl.
    fn update<T: StatefulScalar>(
        parameter: T,
        gradient: T,
        states: [T; 2],
        parameters: &Self::Parameters<T>,
    ) -> StatefulUpdateStep<T>;

    /// Body spelling for launches over scalar `T`: [`Self::BODY`] with its
    /// `{one}` literal replaced by `T`'s width-specific spelling.
    #[must_use]
    fn body_for<T: StatefulScalar>() -> String {
        Self::BODY.replace("{one}", T::ONE_LITERAL)
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
    type Parameters<T: StatefulScalar> = SgdParameters<T>;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "momentum", "padding_zero", "padding_one"];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.momentum + gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * state_zero_next;";

    fn validate_parameters<T: StatefulScalar>(
        parameters: &Self::Parameters<T>,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update<T: StatefulScalar>(
        parameter: T,
        gradient: T,
        states: [T; 2],
        parameters: &Self::Parameters<T>,
    ) -> StatefulUpdateStep<T> {
        let state_zero_next = states[0] * parameters.momentum() + gradient;
        let parameter_next = parameter - parameters.learning_rate() * state_zero_next;
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for Adam {
    type Parameters<T: StatefulScalar> = AdamParameters<T>;
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
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.beta_one + ({one} - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + ({one} - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);";

    fn validate_parameters<T: StatefulScalar>(
        parameters: &Self::Parameters<T>,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update<T: StatefulScalar>(
        parameter: T,
        gradient: T,
        states: [T; 2],
        parameters: &Self::Parameters<T>,
    ) -> StatefulUpdateStep<T> {
        let state_zero_next =
            states[0] * parameters.beta_one() + (T::ONE - parameters.beta_one()) * gradient;
        let state_one_next = states[1] * parameters.beta_two()
            + (T::ONE - parameters.beta_two()) * gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate() * (state_zero_next / parameters.bias_correction_one())
                / ((state_one_next / parameters.bias_correction_two()).sqrt()
                    + parameters.epsilon());
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, state_one_next],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for AdamW {
    type Parameters<T: StatefulScalar> = AdamWParameters<T>;
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
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.beta_one + ({one} - parameters.beta_one) * gradient_value;\n    state_one_next = state_one_value * parameters.beta_two + ({one} - parameters.beta_two) * gradient_value * gradient_value;\n    parameter_next = parameter_value * ({one} - parameters.learning_rate * parameters.weight_decay) - parameters.learning_rate * (state_zero_next / parameters.bias_correction_one) / (sqrt(state_one_next / parameters.bias_correction_two) + parameters.epsilon);";

    fn validate_parameters<T: StatefulScalar>(
        parameters: &Self::Parameters<T>,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update<T: StatefulScalar>(
        parameter: T,
        gradient: T,
        states: [T; 2],
        parameters: &Self::Parameters<T>,
    ) -> StatefulUpdateStep<T> {
        let state_zero_next =
            states[0] * parameters.beta_one() + (T::ONE - parameters.beta_one()) * gradient;
        let state_one_next = states[1] * parameters.beta_two()
            + (T::ONE - parameters.beta_two()) * gradient * gradient;
        let parameter_next = parameter
            * (T::ONE - parameters.learning_rate() * parameters.weight_decay())
            - parameters.learning_rate() * (state_zero_next / parameters.bias_correction_one())
                / ((state_one_next / parameters.bias_correction_two()).sqrt()
                    + parameters.epsilon());
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, state_one_next],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for RmsProp {
    type Parameters<T: StatefulScalar> = RmsPropParameters<T>;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "alpha", "epsilon", "padding"];
    const BODY: &'static str = "state_zero_next = state_zero_value * parameters.alpha + ({one} - parameters.alpha) * gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);";

    fn validate_parameters<T: StatefulScalar>(
        parameters: &Self::Parameters<T>,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update<T: StatefulScalar>(
        parameter: T,
        gradient: T,
        states: [T; 2],
        parameters: &Self::Parameters<T>,
    ) -> StatefulUpdateStep<T> {
        let state_zero_next =
            states[0] * parameters.alpha() + (T::ONE - parameters.alpha()) * gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate() * gradient
                / (state_zero_next.sqrt() + parameters.epsilon());
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }
}

impl<L: KernelDialect> StatefulUpdateRule<L> for AdaGrad {
    type Parameters<T: StatefulScalar> = AdaGradParameters<T>;
    const STATE_COUNT: usize = 1;
    const PARAMETER_FIELDS: &'static [&'static str] =
        &["learning_rate", "epsilon", "padding_zero", "padding_one"];
    const BODY: &'static str = "state_zero_next = state_zero_value + gradient_value * gradient_value;\n    parameter_next = parameter_value - parameters.learning_rate * gradient_value / (sqrt(state_zero_next) + parameters.epsilon);";

    fn validate_parameters<T: StatefulScalar>(
        parameters: &Self::Parameters<T>,
    ) -> super::super::error::Result<()> {
        parameters.validate()
    }

    fn update<T: StatefulScalar>(
        parameter: T,
        gradient: T,
        states: [T; 2],
        parameters: &Self::Parameters<T>,
    ) -> StatefulUpdateStep<T> {
        let state_zero_next = states[0] + gradient * gradient;
        let parameter_next = parameter
            - parameters.learning_rate() * gradient
                / (state_zero_next.sqrt() + parameters.epsilon());
        StatefulUpdateStep {
            parameter: parameter_next,
            states: [state_zero_next, states[1]],
        }
    }
}

#[cfg(test)]
#[path = "tests_rules.rs"]
mod tests;
