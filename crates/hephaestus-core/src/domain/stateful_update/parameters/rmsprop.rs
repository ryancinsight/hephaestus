use eunomia::{Pod, Zeroable};

use super::super::super::error::{HephaestusError, Result};
use super::{require_finite, require_nonnegative};

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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// unless learning rate is finite and non-negative, epsilon is finite
    /// and positive, and alpha lies in `[0, 1)`.
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

/// Validated f64 RMSProp parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RmsPropParametersF64 {
    pub(crate) learning_rate: f64,
    pub(crate) alpha: f64,
    pub(crate) epsilon: f64,
}

impl RmsPropParametersF64 {
    /// Validate and construct f64 RMSProp parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// under the same contract as [`RmsPropParameters::new`](super::RmsPropParameters::new).
    pub fn new(learning_rate: f64, alpha: f64, epsilon: f64) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            alpha,
            epsilon,
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

const _: () = assert!(core::mem::size_of::<RmsPropParameters>() == 16);
const _: () = assert!(core::mem::size_of::<RmsPropParametersF64>() == 24);
