use eunomia::{Pod, Zeroable};

use super::super::super::error::{HephaestusError, Result};
use super::{require_finite, require_nonnegative};

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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// unless learning rate is finite and non-negative and epsilon is
    /// finite and positive.
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

/// Validated f64 AdaGrad parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdaGradParametersF64 {
    pub(crate) learning_rate: f64,
    pub(crate) epsilon: f64,
}

impl AdaGradParametersF64 {
    /// Validate and construct f64 AdaGrad parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// under the same contract as [`AdaGradParameters::new`](super::AdaGradParameters::new).
    pub fn new(learning_rate: f64, epsilon: f64) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            epsilon,
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

const _: () = assert!(core::mem::size_of::<AdaGradParameters>() == 16);
const _: () = assert!(core::mem::size_of::<AdaGradParametersF64>() == 16);
