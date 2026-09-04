use eunomia::{Pod, Zeroable};

use super::super::error::{HephaestusError, Result};

fn require_finite(name: &str, values: &[(&str, f32)]) -> Result<()> {
    if let Some((field, value)) = values.iter().find(|(_, value)| !value.is_finite()) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("{name} parameter {field} must be finite, got {value}"),
        });
    }
    Ok(())
}

fn require_nonnegative(name: &str, field: &str, value: f32) -> Result<()> {
    if value < 0.0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("{name} parameter {field} must be nonnegative, got {value}"),
        });
    }
    Ok(())
}

/// Validated stochastic-gradient-descent parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SgdParameters {
    pub(crate) learning_rate: f32,
    pub(crate) momentum: f32,
    padding: [f32; 2],
}

impl SgdParameters {
    /// Validate and construct SGD parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless the learning
    /// rate is finite and non-negative and momentum is finite in `[0, 1)`.
    pub fn new(learning_rate: f32, momentum: f32) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            momentum,
            padding: [0.0; 2],
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        require_finite(
            "SGD",
            &[
                ("learning_rate", self.learning_rate),
                ("momentum", self.momentum),
            ],
        )?;
        require_nonnegative("SGD", "learning_rate", self.learning_rate)?;
        if !(0.0..1.0).contains(&self.momentum) {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "SGD parameter momentum must lie in [0, 1), got {}",
                    self.momentum
                ),
            });
        }
        Ok(())
    }
}

/// Validated Adam parameters with host-computed bias corrections.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdamParameters {
    pub(crate) learning_rate: f32,
    pub(crate) beta_one: f32,
    pub(crate) beta_two: f32,
    pub(crate) epsilon: f32,
    pub(crate) bias_correction_one: f32,
    pub(crate) bias_correction_two: f32,
    padding: [f32; 2],
}

impl AdamParameters {
    /// Validate and construct Adam parameters for one-based `step`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative, epsilon is finite and positive, beta values
    /// lie in `[0, 1)`, and `step` is positive and representable as `i32`.
    pub fn new(
        learning_rate: f32,
        beta_one: f32,
        beta_two: f32,
        epsilon: f32,
        step: usize,
    ) -> Result<Self> {
        let step = validate_step("Adam", step)?;
        let parameters = Self {
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            bias_correction_one: 1.0 - beta_one.powi(step),
            bias_correction_two: 1.0 - beta_two.powi(step),
            padding: [0.0; 2],
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_adaptive_values(
            "Adam",
            self.learning_rate,
            self.beta_one,
            self.beta_two,
            self.epsilon,
            self.bias_correction_one,
            self.bias_correction_two,
        )
    }
}

/// Validated AdamW parameters with host-computed bias corrections.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdamWParameters {
    pub(crate) learning_rate: f32,
    pub(crate) beta_one: f32,
    pub(crate) beta_two: f32,
    pub(crate) epsilon: f32,
    pub(crate) bias_correction_one: f32,
    pub(crate) bias_correction_two: f32,
    pub(crate) weight_decay: f32,
    padding: f32,
}

impl AdamWParameters {
    /// Validate and construct AdamW parameters for one-based `step`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when the embedded Adam
    /// contract fails or weight decay is not finite and non-negative.
    pub fn new(
        learning_rate: f32,
        beta_one: f32,
        beta_two: f32,
        epsilon: f32,
        weight_decay: f32,
        step: usize,
    ) -> Result<Self> {
        let step = validate_step("AdamW", step)?;
        let parameters = Self {
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            bias_correction_one: 1.0 - beta_one.powi(step),
            bias_correction_two: 1.0 - beta_two.powi(step),
            weight_decay,
            padding: 0.0,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_adaptive_values(
            "AdamW",
            self.learning_rate,
            self.beta_one,
            self.beta_two,
            self.epsilon,
            self.bias_correction_one,
            self.bias_correction_two,
        )?;
        require_finite("AdamW", &[("weight_decay", self.weight_decay)])?;
        require_nonnegative("AdamW", "weight_decay", self.weight_decay)
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

fn validate_adaptive_values(
    name: &str,
    learning_rate: f32,
    beta_one: f32,
    beta_two: f32,
    epsilon: f32,
    bias_correction_one: f32,
    bias_correction_two: f32,
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
    if !(0.0..1.0).contains(&beta_one)
        || !(0.0..1.0).contains(&beta_two)
        || epsilon <= 0.0
        || bias_correction_one <= 0.0
        || bias_correction_two <= 0.0
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
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RmsPropParameters {
    pub(crate) learning_rate: f32,
    pub(crate) alpha: f32,
    pub(crate) epsilon: f32,
    padding: f32,
}

impl RmsPropParameters {
    /// Validate and construct RMSProp parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative, epsilon is finite and positive, and alpha
    /// lies in `[0, 1)`.
    pub fn new(learning_rate: f32, alpha: f32, epsilon: f32) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            alpha,
            epsilon,
            padding: 0.0,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        require_finite(
            "RMSProp",
            &[
                ("learning_rate", self.learning_rate),
                ("alpha", self.alpha),
                ("epsilon", self.epsilon),
            ],
        )?;
        require_nonnegative("RMSProp", "learning_rate", self.learning_rate)?;
        if !(0.0..1.0).contains(&self.alpha) || self.epsilon <= 0.0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: "RMSProp alpha must lie in [0, 1) and epsilon must be positive"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// Validated AdaGrad parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdaGradParameters {
    pub(crate) learning_rate: f32,
    pub(crate) epsilon: f32,
    padding: [f32; 2],
}

impl AdaGradParameters {
    /// Validate and construct AdaGrad parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative and epsilon is finite and positive.
    pub fn new(learning_rate: f32, epsilon: f32) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            epsilon,
            padding: [0.0; 2],
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        require_finite(
            "AdaGrad",
            &[
                ("learning_rate", self.learning_rate),
                ("epsilon", self.epsilon),
            ],
        )?;
        require_nonnegative("AdaGrad", "learning_rate", self.learning_rate)?;
        if self.epsilon <= 0.0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: "AdaGrad epsilon must be positive".to_string(),
            });
        }
        Ok(())
    }
}

const _: () = assert!(core::mem::size_of::<SgdParameters>() == 16);
const _: () = assert!(core::mem::size_of::<AdamParameters>() == 32);
const _: () = assert!(core::mem::size_of::<AdamWParameters>() == 32);
const _: () = assert!(core::mem::size_of::<RmsPropParameters>() == 16);
const _: () = assert!(core::mem::size_of::<AdaGradParameters>() == 16);

#[cfg(test)]
mod tests {
    use eunomia::Zeroable;

    use super::*;
    use crate::{Adam, StatefulUpdateRule, Wgsl};

    fn assert_invalid<T>(result: Result<T>, expected: &str) {
        match result {
            Err(HephaestusError::InvalidConfiguration { message }) => {
                assert_eq!(message, expected)
            }
            Err(error) => panic!("expected InvalidConfiguration({expected:?}), got {error}"),
            Ok(_) => panic!("expected InvalidConfiguration({expected:?}), got success"),
        }
    }

    #[test]
    fn rejects_zeroed_adam_parameter_block() {
        let parameters = AdamParameters::zeroed();
        assert!(<Adam as StatefulUpdateRule<Wgsl>>::validate_parameters(&parameters).is_err());
    }

    #[test]
    fn adam_bias_correction_matches_closed_form() {
        let parameters = AdamParameters::new(0.01, 0.9, 0.999, 1.0e-8, 3)
            .expect("analytically valid Adam parameters");
        assert_eq!(parameters.bias_correction_one, 1.0 - 0.9_f32.powi(3));
        assert_eq!(parameters.bias_correction_two, 1.0 - 0.999_f32.powi(3));
    }

    #[test]
    fn rejects_nonfinite_and_out_of_domain_parameters() {
        assert_invalid(
            SgdParameters::new(f32::NAN, 0.0),
            "SGD parameter learning_rate must be finite, got NaN",
        );
        assert_invalid(
            SgdParameters::new(0.1, 1.0),
            "SGD parameter momentum must lie in [0, 1), got 1",
        );
        assert_invalid(
            AdamParameters::new(-0.1, 0.9, 0.999, 1.0e-8, 1),
            "Adam parameter learning_rate must be nonnegative, got -0.1",
        );
        assert_invalid(
            RmsPropParameters::new(0.1, 1.0, 1.0e-8),
            "RMSProp alpha must lie in [0, 1) and epsilon must be positive",
        );
        assert_invalid(
            AdaGradParameters::new(0.1, 0.0),
            "AdaGrad epsilon must be positive",
        );
        assert_invalid(
            AdamWParameters::new(0.1, 0.9, 0.999, 1.0e-8, -0.1, 1),
            "AdamW parameter weight_decay must be nonnegative, got -0.1",
        );
    }

    #[test]
    fn all_rules_accept_zero_learning_rate() {
        SgdParameters::new(0.0, 0.9).expect("zero-rate SGD parameters");
        AdamParameters::new(0.0, 0.9, 0.999, 1.0e-8, 1).expect("zero-rate Adam parameters");
        AdamWParameters::new(0.0, 0.9, 0.999, 1.0e-8, 0.01, 1).expect("zero-rate AdamW parameters");
        RmsPropParameters::new(0.0, 0.9, 1.0e-8).expect("zero-rate RMSProp parameters");
        AdaGradParameters::new(0.0, 1.0e-8).expect("zero-rate AdaGrad parameters");
    }
}
