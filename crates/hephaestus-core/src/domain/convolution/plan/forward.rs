use leto::{ConvolutionParameters, TransposedConvolutionParameters};

use super::super::ConvolutionForwardOperands;
use super::super::validation::{
    max_forward_offset, reject_aliasing, validate_rank, validate_readonly, validate_writable,
};
use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;

use super::dimensions::regular_dimensions;
use super::dimensions::transposed_dimensions;
use super::plans::ConvolutionPlan;
use super::plans::TransposedConvolutionPlan;

/// Validate a regular convolution forward pass before backend preparation.
///
/// `illegal_aliasing` is the backend's buffer-identity result for output
/// against every readable operand.
///
/// # Errors
///
/// Returns a typed error for rank, shape, storage, output aliasing, arithmetic
/// overflow, or a backend-detected buffer alias.
pub fn plan_convolution_forward<T, B, const R: usize, const S: usize>(
    operands: &ConvolutionForwardOperands<'_, B, R>,
    parameters: ConvolutionParameters<S>,
    illegal_aliasing: bool,
) -> Result<ConvolutionPlan<S>>
where
    B: DeviceBuffer<T>,
{
    validate_rank::<R, S>("convolution")?;
    validate_readonly(operands.input.layout, operands.input.buffer.len())?;
    validate_readonly(operands.weight.layout, operands.weight.buffer.len())?;
    if let Some(bias) = operands.bias {
        validate_readonly(bias.layout, bias.buffer.len())?;
    }
    validate_writable(
        operands.output.layout,
        operands.output.buffer.len(),
        "output",
    )?;
    reject_aliasing(illegal_aliasing)?;

    let mut plan = regular_dimensions(
        operands.input.layout,
        operands.weight.layout,
        operands.bias.map(|view| view.layout),
        operands.output.layout,
        parameters,
    )?;
    plan.max_physical_offset = max_forward_offset(operands)?;
    Ok(plan)
}

/// Validate a transposed convolution forward pass before backend preparation.
///
/// # Errors
///
/// Returns a typed error for rank, shape, storage, output aliasing, arithmetic
/// overflow, or a backend-detected buffer alias.
pub fn plan_transposed_convolution_forward<T, B, const R: usize, const S: usize>(
    operands: &ConvolutionForwardOperands<'_, B, R>,
    parameters: TransposedConvolutionParameters<S>,
    illegal_aliasing: bool,
) -> Result<TransposedConvolutionPlan<S>>
where
    B: DeviceBuffer<T>,
{
    validate_rank::<R, S>("transposed convolution")?;
    validate_readonly(operands.input.layout, operands.input.buffer.len())?;
    validate_readonly(operands.weight.layout, operands.weight.buffer.len())?;
    if let Some(bias) = operands.bias {
        validate_readonly(bias.layout, bias.buffer.len())?;
    }
    validate_writable(
        operands.output.layout,
        operands.output.buffer.len(),
        "output",
    )?;
    reject_aliasing(illegal_aliasing)?;

    let mut plan = transposed_dimensions(
        operands.input.layout,
        operands.weight.layout,
        operands.bias.map(|view| view.layout),
        operands.output.layout,
        parameters,
    )?;
    plan.max_physical_offset = max_forward_offset(operands)?;
    Ok(plan)
}
