use leto::{ConvolutionParameters, Layout, TransposedConvolutionParameters};

use super::super::validation::{
    ChannelDimensions, invalid, validate_channel_shapes, validate_output_spatial,
};
use crate::domain::error::Result;
use crate::domain::planning::map_layout_err;

use super::plans::ConvolutionPlan;
use super::plans::TransposedConvolutionPlan;

pub(crate) fn regular_dimensions<const R: usize, const S: usize>(
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

pub(crate) fn transposed_dimensions<const R: usize, const S: usize>(
    input: &Layout<R>,
    weight: &Layout<R>,
    bias: Option<&Layout<1>>,
    output: &Layout<R>,
    parameters: TransposedConvolutionParameters<S>,
) -> Result<TransposedConvolutionPlan<S>> {
    let batch = input.shape()[0];
    let input_channels = input.shape()[1];
    let output_channels = weight.shape()[1];
    validate_channel_shapes(
        input,
        weight,
        bias,
        output,
        ChannelDimensions::transposed(batch, input_channels, output_channels),
    )?;

    let mut input_spatial = [0; S];
    let mut kernel_spatial = [0; S];
    let mut output_spatial = [0; S];
    for axis in 0..S {
        let input_extent = input.shape()[axis + 2];
        let kernel_extent = weight.shape()[axis + 2];
        if input_extent == 0 || kernel_extent == 0 {
            return Err(invalid(
                "transposed convolution spatial and kernel extents must be nonzero",
            ));
        }
        let expanded_input = (input_extent - 1)
            .checked_mul(parameters.stride()[axis])
            .ok_or_else(|| invalid("transposed convolution expanded input extent overflows"))?;
        let effective_kernel = (kernel_extent - 1)
            .checked_mul(parameters.dilation()[axis])
            .and_then(|extent| extent.checked_add(1))
            .ok_or_else(|| invalid("transposed convolution effective kernel extent overflows"))?;
        let unpadded_output = expanded_input
            .checked_add(effective_kernel)
            .and_then(|extent| extent.checked_add(parameters.output_padding()[axis]))
            .ok_or_else(|| invalid("transposed convolution output extent overflows"))?;
        let total_padding = parameters.padding()[axis]
            .checked_mul(2)
            .ok_or_else(|| invalid("transposed convolution total padding overflows"))?;
        let output_extent = unpadded_output.checked_sub(total_padding).ok_or_else(|| {
            invalid("transposed convolution padding exceeds the generated extent")
        })?;
        input_spatial[axis] = input_extent;
        kernel_spatial[axis] = kernel_extent;
        output_spatial[axis] = output_extent;
    }
    validate_output_spatial(output, &output_spatial)?;
    Ok(TransposedConvolutionPlan {
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
