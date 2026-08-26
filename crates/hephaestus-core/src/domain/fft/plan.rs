use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::planning::map_layout_err;

use super::FftOperands;

/// Dense complex transform direction and its fixed normalization convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FftDirection {
    /// Negative exponential with no scaling.
    Forward,
    /// Positive exponential scaled by the reciprocal of the full element count.
    Inverse,
}

/// Validated dense complex FFT shape and backend address bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FftPlan<const R: usize> {
    /// Transform extents in C-order axis order.
    pub shape: [usize; R],
    /// Product of every transform extent.
    pub elements: usize,
    /// Requested transform direction and normalization.
    pub direction: FftDirection,
    /// Largest physical element offset touched in either component.
    pub max_physical_offset: usize,
}

impl<const R: usize> FftPlan<R> {
    /// Validate every shape and address value narrowed by a backend kernel.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error when a value exceeds the inclusive
    /// backend address limit.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        if self.elements > max_inclusive
            || self.max_physical_offset > max_inclusive
            || self.shape.into_iter().any(|extent| extent > max_inclusive)
        {
            return Err(invalid(format!(
                "FFT plan exceeds backend address limit {max_inclusive}"
            )));
        }
        Ok(())
    }
}

/// Validate split-complex FFT operands before backend preparation.
///
/// `illegal_aliasing` is the backend's buffer-identity result for the real and
/// imaginary components. Core cannot inspect opaque provider handles, so each
/// implementor computes that one identity fact at its boundary.
///
/// # Errors
///
/// Returns a typed rank, shape, storage, layout, overflow, or alias error.
pub fn plan_fft<T, B, const R: usize>(
    operands: &FftOperands<'_, B, R>,
    direction: FftDirection,
    illegal_aliasing: bool,
) -> Result<FftPlan<R>>
where
    B: DeviceBuffer<T>,
{
    if !(1..=3).contains(&R) {
        return Err(invalid(format!(
            "FFT rank {R} is unsupported; expected a rank from 1 through 3"
        )));
    }
    if illegal_aliasing {
        return Err(invalid(
            "FFT real and imaginary components must use distinct buffers",
        ));
    }

    let real = operands.real.layout;
    let imaginary = operands.imaginary.layout;
    if real.shape() != imaginary.shape()
        || real.strides() != imaginary.strides()
        || real.offset() != imaginary.offset()
    {
        return Err(invalid(format!(
            "FFT split-component layouts must match: real shape {:?}, strides {:?}, offset {}; imaginary shape {:?}, strides {:?}, offset {}",
            real.shape(),
            real.strides(),
            real.offset(),
            imaginary.shape(),
            imaginary.strides(),
            imaginary.offset()
        )));
    }
    if real.shape().contains(&0) {
        return Err(invalid("FFT transform extents must be nonzero"));
    }
    if !real.is_c_contiguous() || real.offset() != 0 {
        return Err(invalid(
            "FFT operands must use a zero-offset dense C-order layout",
        ));
    }

    real.validate_storage_len(operands.real.buffer.len())
        .map_err(map_layout_err)?;
    imaginary
        .validate_storage_len(operands.imaginary.buffer.len())
        .map_err(map_layout_err)?;
    let elements = real.checked_size().map_err(map_layout_err)?;
    if operands.real.buffer.len() != elements || operands.imaginary.buffer.len() != elements {
        return Err(invalid(format!(
            "FFT component buffers must contain exactly {elements} elements; real has {}, imaginary has {}",
            operands.real.buffer.len(),
            operands.imaginary.buffer.len()
        )));
    }

    Ok(FftPlan {
        shape: real.shape(),
        elements,
        direction,
        max_physical_offset: elements - 1,
    })
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
