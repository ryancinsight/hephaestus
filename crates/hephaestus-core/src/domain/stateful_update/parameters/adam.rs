use eunomia::{Pod, Zeroable};

use super::super::super::error::Result;
use super::{validate_adaptive_values, validate_step};

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
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// unless learning rate is finite and non-negative, epsilon is finite
    /// and positive, beta values lie in `[0, 1)`, and `step` is positive
    /// and representable as `i32`.
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

/// Validated f64 Adam parameters with host-computed bias corrections.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdamParametersF64 {
    pub(crate) learning_rate: f64,
    pub(crate) beta_one: f64,
    pub(crate) beta_two: f64,
    pub(crate) epsilon: f64,
    pub(crate) bias_correction_one: f64,
    pub(crate) bias_correction_two: f64,
}

impl AdamParametersF64 {
    /// Validate and construct f64 Adam parameters for one-based `step`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`](crate::HephaestusError::InvalidConfiguration)
    /// under the same contract as [`AdamParameters::new`](super::AdamParameters::new).
    pub fn new(
        learning_rate: f64,
        beta_one: f64,
        beta_two: f64,
        epsilon: f64,
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

const _: () = assert!(core::mem::size_of::<AdamParameters>() == 32);
const _: () = assert!(core::mem::size_of::<AdamParametersF64>() == 48);
