use eunomia::{Pod, Zeroable};

use super::super::super::error::{HephaestusError, Result};
use super::{require_finite, require_nonnegative};

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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// unless the learning rate is finite and non-negative and momentum is
    /// finite in `[0, 1)`.
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

/// Validated f64 stochastic-gradient-descent parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SgdParametersF64 {
    pub(crate) learning_rate: f64,
    pub(crate) momentum: f64,
}

impl SgdParametersF64 {
    /// Validate and construct f64 SGD parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// under the same contract as [`SgdParameters::new`](super::SgdParameters::new).
    pub fn new(learning_rate: f64, momentum: f64) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            momentum,
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

const _: () = assert!(core::mem::size_of::<SgdParameters>() == 16);
const _: () = assert!(core::mem::size_of::<SgdParametersF64>() == 16);
