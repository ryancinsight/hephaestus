use leto::Layout;

use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::planning::map_layout_err;

use super::{CrossEntropyBackwardOperands, CrossEntropyForwardOperands};

/// Validated cross-entropy dimensions and backend address bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropyPlan {
    /// Number of independent rows.
    pub batch: usize,
    /// Number of classes in each row.
    pub classes: usize,
    /// Logical logits/probability element count.
    pub elements: usize,
    /// Largest physical element offset touched by any operand.
    pub max_physical_offset: usize,
    /// Gamma-derived tolerance for validating a stored f32 probability row.
    pub probability_tolerance: f32,
}

impl CrossEntropyPlan {
    /// Validate every dimension and offset narrowed by a backend kernel.
    ///
    /// # Errors
    ///
    /// Returns when a value exceeds the inclusive backend address limit.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        if [
            self.batch,
            self.classes,
            self.elements,
            self.max_physical_offset,
        ]
        .into_iter()
        .any(|value| value > max_inclusive)
        {
            return Err(invalid(format!(
                "cross-entropy plan exceeds backend address limit {max_inclusive}"
            )));
        }
        Ok(())
    }
}

/// Validate host-visible forward structure before backend preparation.
///
/// Device preflight remains responsible for target values and finite numeric
/// payloads because those values stay provider-resident.
///
/// # Errors
///
/// Returns a typed shape, storage, layout, overflow, or alias error.
pub fn plan_cross_entropy_forward<T, B, I>(
    operands: &CrossEntropyForwardOperands<'_, B, I>,
    illegal_aliasing: bool,
) -> Result<CrossEntropyPlan>
where
    B: DeviceBuffer<T>,
    I: DeviceBuffer<u32>,
{
    validate_readonly::<T, _, 2>(operands.logits.buffer, operands.logits.layout)?;
    validate_readonly::<u32, _, 1>(operands.targets.buffer, operands.targets.layout)?;
    validate_writable::<T, _, 1>(operands.loss.buffer, operands.loss.layout, "loss")?;
    validate_writable::<T, _, 2>(
        operands.probabilities.buffer,
        operands.probabilities.layout,
        "probabilities",
    )?;
    reject_aliasing(illegal_aliasing)?;

    let [batch, classes] = operands.logits.layout.shape;
    let plan = dimensions(batch, classes)?;
    expect_shape("targets", operands.targets.layout.shape, [batch])?;
    expect_shape("loss", operands.loss.layout.shape, [1])?;
    expect_shape(
        "probabilities",
        operands.probabilities.layout.shape,
        operands.logits.layout.shape,
    )?;

    Ok(CrossEntropyPlan {
        max_physical_offset: [
            max_offset(operands.logits.layout)?,
            max_offset(operands.targets.layout)?,
            max_offset(operands.loss.layout)?,
            max_offset(operands.probabilities.layout)?,
        ]
        .into_iter()
        .max()
        .unwrap_or(0),
        ..plan
    })
}

/// Validate host-visible additive backward structure before preparation.
///
/// # Errors
///
/// Returns a typed shape, storage, layout, overflow, or alias error.
pub fn plan_cross_entropy_backward<T, B, I>(
    operands: &CrossEntropyBackwardOperands<'_, B, I>,
    illegal_aliasing: bool,
) -> Result<CrossEntropyPlan>
where
    B: DeviceBuffer<T>,
    I: DeviceBuffer<u32>,
{
    validate_readonly::<T, _, 1>(
        operands.output_gradient.buffer,
        operands.output_gradient.layout,
    )?;
    validate_readonly::<T, _, 2>(operands.probabilities.buffer, operands.probabilities.layout)?;
    validate_readonly::<u32, _, 1>(operands.targets.buffer, operands.targets.layout)?;
    validate_writable::<T, _, 2>(
        operands.logit_gradient.buffer,
        operands.logit_gradient.layout,
        "logit gradient",
    )?;
    reject_aliasing(illegal_aliasing)?;

    let [batch, classes] = operands.probabilities.layout.shape;
    let plan = dimensions(batch, classes)?;
    expect_shape(
        "output gradient",
        operands.output_gradient.layout.shape,
        [1],
    )?;
    expect_shape("targets", operands.targets.layout.shape, [batch])?;
    expect_shape(
        "logit gradient",
        operands.logit_gradient.layout.shape,
        operands.probabilities.layout.shape,
    )?;

    Ok(CrossEntropyPlan {
        max_physical_offset: [
            max_offset(operands.output_gradient.layout)?,
            max_offset(operands.probabilities.layout)?,
            max_offset(operands.targets.layout)?,
            max_offset(operands.logit_gradient.layout)?,
        ]
        .into_iter()
        .max()
        .unwrap_or(0),
        ..plan
    })
}

fn dimensions(batch: usize, classes: usize) -> Result<CrossEntropyPlan> {
    if batch == 0 {
        return Err(invalid(
            "cross-entropy batch must contain at least one sample",
        ));
    }
    if classes == 0 {
        return Err(invalid(
            "cross-entropy class dimension must contain at least one class",
        ));
    }
    let elements = batch
        .checked_mul(classes)
        .ok_or_else(|| invalid("cross-entropy element count overflows"))?;
    Ok(CrossEntropyPlan {
        batch,
        classes,
        elements,
        max_physical_offset: 0,
        probability_tolerance: probability_tolerance(classes)?,
    })
}

fn probability_tolerance(classes: usize) -> Result<f32> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "f32 kernels represent the runtime class count in native precision"
    )]
    let summation_steps = classes.saturating_sub(1) as f32;
    let summation_error = f32::EPSILON * summation_steps;
    if summation_error >= 1.0 {
        return Err(invalid(format!(
            "cross-entropy class count {classes} exceeds the f32 probability-validation bound"
        )));
    }
    let gamma = summation_error / (1.0 - summation_error);
    let tolerance = gamma + f32::EPSILON * (1.0 + gamma);
    if !tolerance.is_finite() || tolerance >= 0.5 {
        return Err(invalid(format!(
            "cross-entropy class count {classes} exceeds the f32 probability-validation bound"
        )));
    }
    Ok(tolerance)
}

fn validate_readonly<T, B, const R: usize>(buffer: &B, layout: &Layout<R>) -> Result<()>
where
    B: DeviceBuffer<T>,
{
    layout
        .validate_storage_len(buffer.len())
        .map_err(map_layout_err)
}

fn validate_writable<T, B, const R: usize>(buffer: &B, layout: &Layout<R>, name: &str) -> Result<()>
where
    B: DeviceBuffer<T>,
{
    validate_readonly::<T, _, R>(buffer, layout)?;
    if !layout.is_injective().map_err(map_layout_err)? {
        return Err(invalid(format!(
            "cross-entropy {name} layout must map logical indices injectively"
        )));
    }
    Ok(())
}

fn expect_shape<const R: usize>(
    name: &str,
    actual: [usize; R],
    expected: [usize; R],
) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "cross-entropy {name} shape {actual:?} must equal {expected:?}"
        )));
    }
    Ok(())
}

fn max_offset<const R: usize>(layout: &Layout<R>) -> Result<usize> {
    layout
        .checked_min_max_offsets()
        .map(|(_, maximum)| maximum)
        .map_err(map_layout_err)
}

fn reject_aliasing(illegal_aliasing: bool) -> Result<()> {
    if illegal_aliasing {
        return Err(invalid(
            "cross-entropy writable buffers must not alias readable operands or each other",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
