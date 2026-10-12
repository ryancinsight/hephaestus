use eunomia::{FloatElement, Pod, Zeroable};

use super::super::error::{HephaestusError, Result};
use super::scalar::StatefulScalar;

fn require_finite<T: FloatElement + core::fmt::Display>(
    name: &str,
    values: &[(&str, T)],
) -> Result<()> {
    if let Some((field, value)) = values.iter().find(|(_, value)| !value.is_finite()) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("{name} parameter {field} must be finite, got {value}"),
        });
    }
    Ok(())
}

fn require_nonnegative<T: FloatElement + PartialOrd + core::fmt::Display>(
    name: &str,
    field: &str,
    value: T,
) -> Result<()> {
    if value < T::ZERO {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("{name} parameter {field} must be nonnegative, got {value}"),
        });
    }
    Ok(())
}

// Each parameter block is a `repr(transparent)` array of `T` slots in the packed
// uniform order that the rule's field list names. A homogeneous array has no
// inter-field or tail padding, so the block is padding-free at every width.
// The padding slots are explicit, not implicit. `StatefulScalar` is sealed to
// `f32` and `f64`, both `Pod`, so the derived `Pod` and `Zeroable` impls hold
// for every instantiation. The size assertions below pin both widths.

/// Validated stochastic-gradient-descent parameters.
///
/// Slots: learning rate, momentum, two padding slots.
// The bound sits on the definition because the derived `Pod` impl requires
// `Copy` of the slot type for every instantiation, which the derive cannot
// express through use-site bounds alone.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SgdParameters<T: StatefulScalar>([T; 4]);

impl<T: StatefulScalar> SgdParameters<T> {
    /// Validate and construct SGD parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless the learning
    /// rate is finite and non-negative and momentum is finite in `[0, 1)`.
    pub fn new(learning_rate: T, momentum: T) -> Result<Self> {
        let parameters = Self([learning_rate, momentum, T::ZERO, T::ZERO]);
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn learning_rate(&self) -> T {
        self.0[0]
    }

    pub(crate) fn momentum(&self) -> T {
        self.0[1]
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let (learning_rate, momentum) = (self.learning_rate(), self.momentum());
        require_finite(
            "SGD",
            &[("learning_rate", learning_rate), ("momentum", momentum)],
        )?;
        require_nonnegative("SGD", "learning_rate", learning_rate)?;
        if !(T::ZERO..T::ONE).contains(&momentum) {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("SGD parameter momentum must lie in [0, 1), got {momentum}"),
            });
        }
        Ok(())
    }
}

/// Validated Adam parameters with host-computed bias corrections.
///
/// Slots: learning rate, `beta_one`, `beta_two`, epsilon, the two bias
/// corrections, two padding slots.
// The bound sits on the definition because the derived `Pod` impl requires
// `Copy` of the slot type for every instantiation, which the derive cannot
// express through use-site bounds alone.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdamParameters<T: StatefulScalar>([T; 8]);

impl<T: StatefulScalar> AdamParameters<T> {
    /// Validate and construct Adam parameters for one-based `step`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative, epsilon is finite and positive, beta values
    /// lie in `[0, 1)`, and `step` is positive and representable as `i32`.
    pub fn new(
        learning_rate: T,
        beta_one: T,
        beta_two: T,
        epsilon: T,
        step: usize,
    ) -> Result<Self> {
        let step = validate_step("Adam", step)?;
        let parameters = Self([
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            T::ONE - beta_one.integer_power(step),
            T::ONE - beta_two.integer_power(step),
            T::ZERO,
            T::ZERO,
        ]);
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn learning_rate(&self) -> T {
        self.0[0]
    }

    pub(crate) fn beta_one(&self) -> T {
        self.0[1]
    }

    pub(crate) fn beta_two(&self) -> T {
        self.0[2]
    }

    pub(crate) fn epsilon(&self) -> T {
        self.0[3]
    }

    pub(crate) fn bias_correction_one(&self) -> T {
        self.0[4]
    }

    pub(crate) fn bias_correction_two(&self) -> T {
        self.0[5]
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_adaptive_values(
            "Adam",
            self.learning_rate(),
            self.beta_one(),
            self.beta_two(),
            self.epsilon(),
            self.bias_correction_one(),
            self.bias_correction_two(),
        )
    }
}

/// Validated AdamW parameters with host-computed bias corrections.
///
/// Slots: the Adam slots, then weight decay, then one padding slot.
// The bound sits on the definition because the derived `Pod` impl requires
// `Copy` of the slot type for every instantiation, which the derive cannot
// express through use-site bounds alone.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdamWParameters<T: StatefulScalar>([T; 8]);

impl<T: StatefulScalar> AdamWParameters<T> {
    /// Validate and construct AdamW parameters for one-based `step`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when the embedded Adam
    /// contract fails or weight decay is not finite and non-negative.
    pub fn new(
        learning_rate: T,
        beta_one: T,
        beta_two: T,
        epsilon: T,
        weight_decay: T,
        step: usize,
    ) -> Result<Self> {
        let step = validate_step("AdamW", step)?;
        let parameters = Self([
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            T::ONE - beta_one.integer_power(step),
            T::ONE - beta_two.integer_power(step),
            weight_decay,
            T::ZERO,
        ]);
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn learning_rate(&self) -> T {
        self.0[0]
    }

    pub(crate) fn beta_one(&self) -> T {
        self.0[1]
    }

    pub(crate) fn beta_two(&self) -> T {
        self.0[2]
    }

    pub(crate) fn epsilon(&self) -> T {
        self.0[3]
    }

    pub(crate) fn bias_correction_one(&self) -> T {
        self.0[4]
    }

    pub(crate) fn bias_correction_two(&self) -> T {
        self.0[5]
    }

    pub(crate) fn weight_decay(&self) -> T {
        self.0[6]
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_adaptive_values(
            "AdamW",
            self.learning_rate(),
            self.beta_one(),
            self.beta_two(),
            self.epsilon(),
            self.bias_correction_one(),
            self.bias_correction_two(),
        )?;
        require_finite("AdamW", &[("weight_decay", self.weight_decay())])?;
        require_nonnegative("AdamW", "weight_decay", self.weight_decay())
    }
}

fn validate_step(name: &str, step: usize) -> Result<i32> {
    if step == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("{name} step must be nonzero"),
        });
    }
    i32::try_from(step).map_err(|_| HephaestusError::InvalidConfiguration {
        message: format!("{name} step {step} exceeds i32 range"),
    })
}

fn validate_adaptive_values<T: StatefulScalar>(
    name: &str,
    learning_rate: T,
    beta_one: T,
    beta_two: T,
    epsilon: T,
    bias_correction_one: T,
    bias_correction_two: T,
) -> Result<()> {
    require_finite(
        name,
        &[
            ("learning_rate", learning_rate),
            ("beta_one", beta_one),
            ("beta_two", beta_two),
            ("epsilon", epsilon),
            ("bias_correction_one", bias_correction_one),
            ("bias_correction_two", bias_correction_two),
        ],
    )?;
    require_nonnegative(name, "learning_rate", learning_rate)?;
    if !(T::ZERO..T::ONE).contains(&beta_one)
        || !(T::ZERO..T::ONE).contains(&beta_two)
        || epsilon <= T::ZERO
        || bias_correction_one <= T::ZERO
        || bias_correction_two <= T::ZERO
    {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "{name} beta values must lie in [0, 1), epsilon and bias corrections must be positive"
            ),
        });
    }
    Ok(())
}

/// Validated RMSProp parameters.
///
/// Slots: learning rate, alpha, epsilon, one padding slot.
// The bound sits on the definition because the derived `Pod` impl requires
// `Copy` of the slot type for every instantiation, which the derive cannot
// express through use-site bounds alone.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RmsPropParameters<T: StatefulScalar>([T; 4]);

impl<T: StatefulScalar> RmsPropParameters<T> {
    /// Validate and construct RMSProp parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative, epsilon is finite and positive, and alpha
    /// lies in `[0, 1)`.
    pub fn new(learning_rate: T, alpha: T, epsilon: T) -> Result<Self> {
        let parameters = Self([learning_rate, alpha, epsilon, T::ZERO]);
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn learning_rate(&self) -> T {
        self.0[0]
    }

    pub(crate) fn alpha(&self) -> T {
        self.0[1]
    }

    pub(crate) fn epsilon(&self) -> T {
        self.0[2]
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let (learning_rate, alpha, epsilon) = (self.learning_rate(), self.alpha(), self.epsilon());
        require_finite(
            "RMSProp",
            &[
                ("learning_rate", learning_rate),
                ("alpha", alpha),
                ("epsilon", epsilon),
            ],
        )?;
        require_nonnegative("RMSProp", "learning_rate", learning_rate)?;
        if !(T::ZERO..T::ONE).contains(&alpha) || epsilon <= T::ZERO {
            return Err(HephaestusError::InvalidConfiguration {
                message: "RMSProp alpha must lie in [0, 1) and epsilon must be positive"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// Validated AdaGrad parameters.
///
/// Slots: learning rate, epsilon, two padding slots.
// The bound sits on the definition because the derived `Pod` impl requires
// `Copy` of the slot type for every instantiation, which the derive cannot
// express through use-site bounds alone.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdaGradParameters<T: StatefulScalar>([T; 4]);

impl<T: StatefulScalar> AdaGradParameters<T> {
    /// Validate and construct AdaGrad parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative and epsilon is finite and positive.
    pub fn new(learning_rate: T, epsilon: T) -> Result<Self> {
        let parameters = Self([learning_rate, epsilon, T::ZERO, T::ZERO]);
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn learning_rate(&self) -> T {
        self.0[0]
    }

    pub(crate) fn epsilon(&self) -> T {
        self.0[1]
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let (learning_rate, epsilon) = (self.learning_rate(), self.epsilon());
        require_finite(
            "AdaGrad",
            &[("learning_rate", learning_rate), ("epsilon", epsilon)],
        )?;
        require_nonnegative("AdaGrad", "learning_rate", learning_rate)?;
        if epsilon <= T::ZERO {
            return Err(HephaestusError::InvalidConfiguration {
                message: "AdaGrad epsilon must be positive".to_string(),
            });
        }
        Ok(())
    }
}

// The padded slot counts equal the rule field lists at both widths. The f32
// sizes are the uniform sizes the pre-generic structs fixed; the f64 sizes
// double them with the same slot counts.
const _: () = assert!(core::mem::size_of::<SgdParameters<f32>>() == 16);
const _: () = assert!(core::mem::size_of::<SgdParameters<f64>>() == 32);
const _: () = assert!(core::mem::size_of::<AdamParameters<f32>>() == 32);
const _: () = assert!(core::mem::size_of::<AdamParameters<f64>>() == 64);
const _: () = assert!(core::mem::size_of::<AdamWParameters<f32>>() == 32);
const _: () = assert!(core::mem::size_of::<AdamWParameters<f64>>() == 64);
const _: () = assert!(core::mem::size_of::<RmsPropParameters<f32>>() == 16);
const _: () = assert!(core::mem::size_of::<RmsPropParameters<f64>>() == 32);
const _: () = assert!(core::mem::size_of::<AdaGradParameters<f32>>() == 16);
const _: () = assert!(core::mem::size_of::<AdaGradParameters<f64>>() == 32);

#[cfg(test)]
#[path = "tests_parameters.rs"]
mod tests;
