mod adagrad;
mod adam;
mod adamw;
mod rmsprop;
mod sgd;
#[cfg(test)]
mod tests;

pub use adagrad::{AdaGradParameters, AdaGradParametersF64};
pub use adam::{AdamParameters, AdamParametersF64};
pub use adamw::{AdamWParameters, AdamWParametersF64};
pub use rmsprop::{RmsPropParameters, RmsPropParametersF64};
pub use sgd::{SgdParameters, SgdParametersF64};

use eunomia::FloatElement;

use super::super::error::{HephaestusError, Result};

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

// f64 uniform twins. The f32 structs are a frozen uniform ABI (their
// padding fields keep the 16/32-byte sizes device kernels were built
// against), so f64 gets its own padding-free structs — field counts times
// 8 bytes already satisfy uniform granularity — sharing the generic
// validation helpers with the f32 originals.
