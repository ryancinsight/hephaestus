use leto::{ConvolutionParameters, TransposedConvolutionParameters};

use super::super::ConvolutionBackwardOperands;
use super::super::validation::{
    invalid, max_backward_offset, reject_aliasing, validate_bias_gradient,
    validate_gradient_targets, validate_rank, validate_readonly,
};
use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;

use super::dimensions::regular_dimensions;
use super::dimensions::transposed_dimensions;
use super::plans::ConvolutionPlan;
use super::plans::TransposedConvolutionPlan;

/// Validate a regular convolution additive backward pass.
///
/// `illegal_aliasing` is the backend's aggregate buffer-identity result for
/// every selected gradient against readable operands and other targets.
///
/// # Errors
///
/// Returns a typed error for an empty target set, rank, shape, storage,
/// writable-layout aliasing, arithmetic overflow, or buffer aliasing.
pub fn plan_convolution_backward<T, B, const R: usize, const S: usize>(
    operands: &ConvolutionBackwardOperands<'_, B, R>,
    parameters: ConvolutionParameters<S>,
    illegal_aliasing: bool,
) -> Result<ConvolutionPlan<S>>
where
    B: DeviceBuffer<T>,
{
    validate_rank::<R, S>("convolution")?;
    if operands.gradients.is_empty() {
        return Err(invalid(
            "convolution backward requires at least one gradient target",
        ));
    }
    validate_readonly(operands.input.layout, operands.input.buffer.len())?;
    validate_readonly(operands.weight.layout, operands.weight.buffer.len())?;
    validate_readonly(
        operands.grad_output.layout,
        operands.grad_output.buffer.len(),
    )?;
    validate_gradient_targets(
        operands,
        operands.input.layout.shape(),
        operands.weight.layout.shape(),
    )?;
    reject_aliasing(illegal_aliasing)?;

    let mut plan = regular_dimensions(
        operands.input.layout,
        operands.weight.layout,
        None,
        operands.grad_output.layout,
        parameters,
    )?;
    validate_bias_gradient(
        operands.gradients.bias.map(|view| view.layout),
        plan.output_channels,
    )?;
    plan.max_physical_offset = max_backward_offset(operands)?;
    Ok(plan)
}

/// Validate a transposed convolution additive backward pass.
///
/// # Errors
///
/// Returns a typed error for an empty target set, rank, shape, storage,
/// writable-layout aliasing, arithmetic overflow, or buffer aliasing.
pub fn plan_transposed_convolution_backward<T, B, const R: usize, const S: usize>(
    operands: &ConvolutionBackwardOperands<'_, B, R>,
    parameters: TransposedConvolutionParameters<S>,
    illegal_aliasing: bool,
) -> Result<TransposedConvolutionPlan<S>>
where
    B: DeviceBuffer<T>,
{
    validate_rank::<R, S>("transposed convolution")?;
    if operands.gradients.is_empty() {
        return Err(invalid(
            "transposed convolution backward requires at least one gradient target",
        ));
    }
    validate_readonly(operands.input.layout, operands.input.buffer.len())?;
    validate_readonly(operands.weight.layout, operands.weight.buffer.len())?;
    validate_readonly(
        operands.grad_output.layout,
        operands.grad_output.buffer.len(),
    )?;
    validate_gradient_targets(
        operands,
        operands.input.layout.shape(),
        operands.weight.layout.shape(),
    )?;
    reject_aliasing(illegal_aliasing)?;

    let mut plan = transposed_dimensions(
        operands.input.layout,
        operands.weight.layout,
        None,
        operands.grad_output.layout,
        parameters,
    )?;
    validate_bias_gradient(
        operands.gradients.bias.map(|view| view.layout),
        plan.output_channels,
    )?;
    plan.max_physical_offset = max_backward_offset(operands)?;
    Ok(plan)
}
