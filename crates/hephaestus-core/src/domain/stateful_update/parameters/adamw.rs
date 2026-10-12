use eunomia::{Pod, Zeroable};

use super::super::super::error::Result;
use super::{require_finite, require_nonnegative, validate_adaptive_values, validate_step};

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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// when the embedded Adam contract fails or weight decay is not finite
    /// and non-negative.
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

/// Validated f64 AdamW parameters with host-computed bias corrections.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdamWParametersF64 {
    pub(crate) learning_rate: f64,
    pub(crate) beta_one: f64,
    pub(crate) beta_two: f64,
    pub(crate) epsilon: f64,
    pub(crate) bias_correction_one: f64,
    pub(crate) bias_correction_two: f64,
    pub(crate) weight_decay: f64,
}

impl AdamWParametersF64 {
    /// Validate and construct f64 AdamW parameters for one-based `step`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// under the same contract as [`AdamWParameters::new`](super::AdamWParameters::new).
    pub fn new(
        learning_rate: f64,
        beta_one: f64,
        beta_two: f64,
        epsilon: f64,
        weight_decay: f64,
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

const _: () = assert!(core::mem::size_of::<AdamWParameters>() == 32);
const _: () = assert!(core::mem::size_of::<AdamWParametersF64>() == 56);
