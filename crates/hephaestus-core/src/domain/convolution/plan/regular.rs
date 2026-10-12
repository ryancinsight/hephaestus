use leto::{ConvolutionParameters, Layout};

use super::super::validation::{
    ChannelDimensions, invalid, max_backward_offset, max_forward_offset, reject_aliasing,
    validate_bias_gradient, validate_channel_shapes, validate_gradient_targets,
    validate_output_spatial, validate_rank, validate_readonly, validate_writable,
};
use super::super::{ConvolutionBackwardOperands, ConvolutionForwardOperands};
use super::limits::{validate_plan_address_limit, validate_regular_projection_limit};
use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;
use crate::domain::planning::map_layout_err;

/// Validated regular-convolution dimensions and launch bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvolutionPlan<const S: usize> {
    /// Batch extent.
    pub batch: usize,
    /// Input channel extent.
    pub input_channels: usize,
    /// Output channel extent.
    pub output_channels: usize,
    /// Input spatial extents.
    pub input_spatial: [usize; S],
    /// Kernel spatial extents.
    pub kernel_spatial: [usize; S],
    /// Output spatial extents.
    pub output_spatial: [usize; S],
    /// Validated convolution parameters.
    pub parameters: ConvolutionParameters<S>,
    /// Logical output element count.
    pub output_elements: usize,
    /// Largest physical element offset touched by any operand.
    pub max_physical_offset: usize,
}

impl<const S: usize> ConvolutionPlan<S> {
    /// Validate every value narrowed by a backend kernel's address contract.
    ///
    /// Backends pass the inclusive limit of every integer representation used
    /// by their address and projection arithmetic. CUDA and the current WGSL
    /// implementation both use signed 32-bit projection arithmetic and
    /// therefore pass `i32::MAX`.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error when any extent, parameter, logical
    /// element count, or physical offset exceeds `max_inclusive`.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        validate_plan_address_limit(
            max_inclusive,
            [
                self.batch,
                self.input_channels,
                self.output_channels,
                self.output_elements,
                self.max_physical_offset,
            ],
            [
                &self.input_spatial,
                &self.kernel_spatial,
                &self.output_spatial,
                self.parameters.stride(),
                self.parameters.padding(),
                self.parameters.dilation(),
            ],
        )?;
        validate_regular_projection_limit(self, max_inclusive)
    }
}

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

fn regular_dimensions<const R: usize, const S: usize>(
    input: &Layout<R>,
    weight: &Layout<R>,
    bias: Option<&Layout<1>>,
    output: &Layout<R>,
    parameters: ConvolutionParameters<S>,
) -> Result<ConvolutionPlan<S>> {
    let batch = input.shape()[0];
    let input_channels = input.shape()[1];
    let output_channels = weight.shape()[0];
    validate_channel_shapes(
        input,
        weight,
        bias,
        output,
        ChannelDimensions::regular(batch, input_channels, output_channels),
    )?;

    let mut input_spatial = [0; S];
    let mut kernel_spatial = [0; S];
    let mut output_spatial = [0; S];
    for axis in 0..S {
        let input_extent = input.shape()[axis + 2];
        let kernel_extent = weight.shape()[axis + 2];
        if kernel_extent == 0 {
            return Err(invalid("convolution kernel extents must be nonzero"));
        }
        let effective_kernel = parameters.dilation()[axis]
            .checked_mul(kernel_extent - 1)
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| invalid("convolution effective kernel extent overflows"))?;
        let padded_input = parameters.padding()[axis]
            .checked_mul(2)
            .and_then(|padding| input_extent.checked_add(padding))
            .ok_or_else(|| invalid("convolution padded input extent overflows"))?;
        let output_extent = padded_input
            .checked_sub(effective_kernel)
            .map_or(0, |extent| extent / parameters.stride()[axis] + 1);
        input_spatial[axis] = input_extent;
        kernel_spatial[axis] = kernel_extent;
        output_spatial[axis] = output_extent;
    }
    validate_output_spatial(output, &output_spatial)?;
    Ok(ConvolutionPlan {
        batch,
        input_channels,
        output_channels,
        input_spatial,
        kernel_spatial,
        output_spatial,
        parameters,
        output_elements: output.checked_size().map_err(map_layout_err)?,
        max_physical_offset: 0,
    })
}
