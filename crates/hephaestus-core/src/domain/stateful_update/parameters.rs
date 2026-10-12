use eunomia::{FloatElement, Pod, Zeroable};

use super::super::error::{HephaestusError, Result};
use super::width::UpdateWidth;

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

/// Validated SGD parameters.
///
/// Generic over the update width: `SgdParameters` is the f32 spelling (the
/// frozen 16-byte uniform block device kernels were built against),
/// `SgdParameters<f64>` the padding-free double-precision block. Layouts are
/// pinned by the size asserts below.
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct SgdParameters<T: UpdateWidth = f32> {
    pub(crate) learning_rate: T,
    pub(crate) momentum: T,
    padding: T::Pad<2>,
}

// SAFETY: `repr(C)` with every field `Pod` at identical alignment, so no
// padding bytes exist and every bit pattern is valid. Layout pinned by the
// size asserts below. (`Pod` cannot be derived: the derive rejects generic
// `repr(C)` structs on stable Rust.)
#[expect(unsafe_code, reason = "proven padding-free Pod impl; see SAFETY above")]
unsafe impl<T: UpdateWidth> Pod for SgdParameters<T> {}

impl<T: UpdateWidth> SgdParameters<T> {
    /// Validate and construct SGD parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// and momentum are finite and momentum lies in `[0, 1)`.
    pub fn new(learning_rate: T, momentum: T) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            momentum,
            padding: T::Pad::<2>::zeroed(),
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
        if !(T::ZERO..T::ONE).contains(&self.momentum) {
            return Err(HephaestusError::InvalidConfiguration {
                message: "SGD momentum must lie in [0, 1)".to_string(),
            });
        }
        Ok(())
    }
}

/// Validated Adam parameters with host-computed bias corrections.
///
/// Generic over the update width: `AdamParameters` is the f32 spelling (the
/// frozen 32-byte uniform block), `AdamParameters<f64>` the padding-free
/// double-precision block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct AdamParameters<T: UpdateWidth = f32> {
    pub(crate) learning_rate: T,
    pub(crate) beta_one: T,
    pub(crate) beta_two: T,
    pub(crate) epsilon: T,
    pub(crate) bias_correction_one: T,
    pub(crate) bias_correction_two: T,
    padding: T::Pad<2>,
}

// SAFETY: `repr(C)` with every field `Pod` at identical alignment, so no
// padding bytes exist and every bit pattern is valid. Layout pinned by the
// size asserts below.
#[expect(unsafe_code, reason = "proven padding-free Pod impl; see SAFETY above")]
unsafe impl<T: UpdateWidth> Pod for AdamParameters<T> {}

impl<T: UpdateWidth> AdamParameters<T> {
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
        let parameters = Self {
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            bias_correction_one: T::ONE - beta_one.powi(step),
            bias_correction_two: T::ONE - beta_two.powi(step),
            padding: T::Pad::<2>::zeroed(),
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
///
/// Generic over the update width: `AdamWParameters` is the f32 spelling (the
/// frozen 32-byte uniform block), `AdamWParameters<f64>` the padding-free
/// double-precision block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct AdamWParameters<T: UpdateWidth = f32> {
    pub(crate) learning_rate: T,
    pub(crate) beta_one: T,
    pub(crate) beta_two: T,
    pub(crate) epsilon: T,
    pub(crate) bias_correction_one: T,
    pub(crate) bias_correction_two: T,
    pub(crate) weight_decay: T,
    padding: T::Pad<1>,
}

// SAFETY: `repr(C)` with every field `Pod` at identical alignment, so no
// padding bytes exist and every bit pattern is valid. Layout pinned by the
// size asserts below.
#[expect(unsafe_code, reason = "proven padding-free Pod impl; see SAFETY above")]
unsafe impl<T: UpdateWidth> Pod for AdamWParameters<T> {}

impl<T: UpdateWidth> AdamWParameters<T> {
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
        let parameters = Self {
            learning_rate,
            beta_one,
            beta_two,
            epsilon,
            bias_correction_one: T::ONE - beta_one.powi(step),
            bias_correction_two: T::ONE - beta_two.powi(step),
            weight_decay,
            padding: T::Pad::<1>::zeroed(),
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

fn validate_adaptive_values<T: FloatElement + PartialOrd + core::fmt::Display>(
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
/// Generic over the update width: `RmsPropParameters` is the f32 spelling
/// (the frozen 16-byte uniform block), `RmsPropParameters<f64>` the
/// padding-free double-precision block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct RmsPropParameters<T: UpdateWidth = f32> {
    pub(crate) learning_rate: T,
    pub(crate) alpha: T,
    pub(crate) epsilon: T,
    padding: T::Pad<1>,
}

// SAFETY: `repr(C)` with every field `Pod` at identical alignment, so no
// padding bytes exist and every bit pattern is valid. Layout pinned by the
// size asserts below.
#[expect(unsafe_code, reason = "proven padding-free Pod impl; see SAFETY above")]
unsafe impl<T: UpdateWidth> Pod for RmsPropParameters<T> {}

impl<T: UpdateWidth> RmsPropParameters<T> {
    /// Validate and construct RMSProp parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative, epsilon is finite and positive, and alpha
    /// lies in `[0, 1)`.
    pub fn new(learning_rate: T, alpha: T, epsilon: T) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            alpha,
            epsilon,
            padding: T::Pad::<1>::zeroed(),
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
        if !(T::ZERO..T::ONE).contains(&self.alpha) || self.epsilon <= T::ZERO {
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
/// Generic over the update width: `AdaGradParameters` is the f32 spelling
/// (the frozen 16-byte uniform block), `AdaGradParameters<f64>` the
/// padding-free double-precision block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct AdaGradParameters<T: UpdateWidth = f32> {
    pub(crate) learning_rate: T,
    pub(crate) epsilon: T,
    padding: T::Pad<2>,
}

// SAFETY: `repr(C)` with every field `Pod` at identical alignment, so no
// padding bytes exist and every bit pattern is valid. Layout pinned by the
// size asserts below.
#[expect(unsafe_code, reason = "proven padding-free Pod impl; see SAFETY above")]
unsafe impl<T: UpdateWidth> Pod for AdaGradParameters<T> {}

impl<T: UpdateWidth> AdaGradParameters<T> {
    /// Validate and construct AdaGrad parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] unless learning rate
    /// is finite and non-negative and epsilon is finite and positive.
    pub fn new(learning_rate: T, epsilon: T) -> Result<Self> {
        let parameters = Self {
            learning_rate,
            epsilon,
            padding: T::Pad::<2>::zeroed(),
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
        if self.epsilon <= T::ZERO {
            return Err(HephaestusError::InvalidConfiguration {
                message: "AdaGrad epsilon must be positive".to_string(),
            });
        }
        Ok(())
    }
}

// The f32 layouts above are a FROZEN UNIFORM ABI (their padding fields keep
// the sizes device kernels were built against); the f64 layouts are the
// padding-free twins. Both monomorphizations are pinned here.
const _: () = assert!(core::mem::size_of::<SgdParameters<f32>>() == 16);
const _: () = assert!(core::mem::size_of::<AdamParameters<f32>>() == 32);
const _: () = assert!(core::mem::size_of::<AdamWParameters<f32>>() == 32);
const _: () = assert!(core::mem::size_of::<RmsPropParameters<f32>>() == 16);
const _: () = assert!(core::mem::size_of::<AdaGradParameters<f32>>() == 16);
const _: () = assert!(core::mem::size_of::<SgdParameters<f64>>() == 16);
const _: () = assert!(core::mem::size_of::<AdamParameters<f64>>() == 48);
const _: () = assert!(core::mem::size_of::<AdamWParameters<f64>>() == 56);
const _: () = assert!(core::mem::size_of::<RmsPropParameters<f64>>() == 24);
const _: () = assert!(core::mem::size_of::<AdaGradParameters<f64>>() == 16);

#[cfg(test)]
#[path = "tests_parameters.rs"]
mod tests;
