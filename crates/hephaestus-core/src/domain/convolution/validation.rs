use leto::Layout;

use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::planning::map_layout_err;

use super::{ConvolutionBackwardOperands, ConvolutionForwardOperands};

#[derive(Clone, Copy)]
enum WeightChannelOrder {
    OutputThenInput,
    InputThenOutput,
}

#[derive(Clone, Copy)]
pub(super) struct ChannelDimensions {
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    weight_order: WeightChannelOrder,
}

impl ChannelDimensions {
    pub(super) const fn regular(
        batch: usize,
        input_channels: usize,
        output_channels: usize,
    ) -> Self {
        Self {
            batch,
            input_channels,
            output_channels,
            weight_order: WeightChannelOrder::OutputThenInput,
        }
    }

    pub(super) const fn transposed(
        batch: usize,
        input_channels: usize,
        output_channels: usize,
    ) -> Self {
        Self {
            batch,
            input_channels,
            output_channels,
            weight_order: WeightChannelOrder::InputThenOutput,
        }
    }
}

pub(super) fn validate_channel_shapes<const R: usize>(
    input: &Layout<R>,
    weight: &Layout<R>,
    bias: Option<&Layout<1>>,
    output: &Layout<R>,
    dimensions: ChannelDimensions,
) -> Result<()> {
    let weight_channels_match = match dimensions.weight_order {
        WeightChannelOrder::OutputThenInput => weight.shape[1] == dimensions.input_channels,
        WeightChannelOrder::InputThenOutput => weight.shape[0] == dimensions.input_channels,
    };
    if !weight_channels_match
        || output.shape[0] != dimensions.batch
        || output.shape[1] != dimensions.output_channels
    {
        return Err(invalid(format!(
            "convolution channel shape mismatch: input {:?}, weight {:?}, output {:?}",
            input.shape, weight.shape, output.shape
        )));
    }
    if let Some(bias) = bias {
        if bias.shape != [dimensions.output_channels] {
            return Err(invalid(format!(
                "convolution bias shape {:?} must equal [{}]",
                bias.shape, dimensions.output_channels
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_rank<const R: usize, const S: usize>(name: &str) -> Result<()> {
    let expected = S
        .checked_add(2)
        .ok_or_else(|| invalid(format!("{name} tensor rank overflows")))?;
    if S == 0 || R != expected {
        return Err(invalid(format!(
            "{name} tensor rank {R} must equal spatial rank {S} plus batch/channel axes"
        )));
    }
    Ok(())
}

pub(super) fn validate_output_spatial<const R: usize, const S: usize>(
    output: &Layout<R>,
    expected: &[usize; S],
) -> Result<()> {
    if output.shape[2..] != *expected {
        return Err(invalid(format!(
            "convolution output spatial shape {:?} must equal {expected:?}",
            &output.shape[2..]
        )));
    }
    Ok(())
}

pub(super) fn validate_readonly<const R: usize>(layout: &Layout<R>, len: usize) -> Result<()> {
    layout.validate_storage_len(len).map_err(map_layout_err)
}

pub(super) fn validate_writable<const R: usize>(
    layout: &Layout<R>,
    len: usize,
    name: &str,
) -> Result<()> {
    validate_readonly(layout, len)?;
    if !writable_layout_is_nonoverlapping(layout)? {
        return Err(invalid(format!(
            "convolution {name} layout must be non-overlapping"
        )));
    }
    Ok(())
}

fn writable_layout_is_nonoverlapping<const R: usize>(layout: &Layout<R>) -> Result<bool> {
    if layout.checked_size().map_err(map_layout_err)? == 0 {
        return Ok(true);
    }

    let mut axes = [(0_usize, 0_usize); R];
    let mut active = 0;
    for (&extent, &stride) in layout.shape.iter().zip(&layout.strides) {
        if extent <= 1 {
            continue;
        }
        let magnitude = stride.unsigned_abs();
        if magnitude == 0 {
            return Ok(false);
        }
        axes[active] = (magnitude, extent);
        active += 1;
    }
    axes[..active].sort_unstable_by_key(|&(stride, _)| stride);

    let mut covered_span = 1_usize;
    for &(stride, extent) in &axes[..active] {
        if stride < covered_span {
            return Ok(false);
        }
        covered_span = (extent - 1)
            .checked_mul(stride)
            .and_then(|axis_span| covered_span.checked_add(axis_span))
            .ok_or_else(|| invalid("convolution writable layout span overflows"))?;
    }
    Ok(true)
}

pub(super) fn validate_gradient_targets<T, B, const R: usize>(
    operands: &ConvolutionBackwardOperands<'_, B, R>,
    input_shape: [usize; R],
    weight_shape: [usize; R],
) -> Result<()>
where
    B: DeviceBuffer<T>,
{
    if let Some(target) = operands.gradients.input {
        validate_writable(target.layout, target.buffer.len(), "input gradient")?;
        validate_target_shape(target.layout.shape, input_shape, "input gradient")?;
    }
    if let Some(target) = operands.gradients.weight {
        validate_writable(target.layout, target.buffer.len(), "weight gradient")?;
        validate_target_shape(target.layout.shape, weight_shape, "weight gradient")?;
    }
    if let Some(target) = operands.gradients.bias {
        validate_writable(target.layout, target.buffer.len(), "bias gradient")?;
    }
    Ok(())
}

pub(super) fn validate_bias_gradient(
    layout: Option<&Layout<1>>,
    output_channels: usize,
) -> Result<()> {
    if let Some(layout) = layout {
        validate_target_shape(layout.shape, [output_channels], "bias gradient")?;
    }
    Ok(())
}

fn validate_target_shape<const R: usize>(
    actual: [usize; R],
    expected: [usize; R],
    name: &str,
) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "convolution {name} shape {actual:?} must equal {expected:?}"
        )));
    }
    Ok(())
}

pub(super) fn reject_aliasing(illegal_aliasing: bool) -> Result<()> {
    if illegal_aliasing {
        return Err(invalid(
            "convolution writable buffers must not alias readable operands or each other",
        ));
    }
    Ok(())
}

pub(super) fn max_forward_offset<T, B, const R: usize>(
    operands: &ConvolutionForwardOperands<'_, B, R>,
) -> Result<usize>
where
    B: DeviceBuffer<T>,
{
    let mut maximum = max_offset(operands.input.layout)?;
    maximum = maximum.max(max_offset(operands.weight.layout)?);
    maximum = maximum.max(max_offset(operands.output.layout)?);
    if let Some(bias) = operands.bias {
        maximum = maximum.max(max_offset(bias.layout)?);
    }
    Ok(maximum)
}

pub(super) fn max_backward_offset<T, B, const R: usize>(
    operands: &ConvolutionBackwardOperands<'_, B, R>,
) -> Result<usize>
where
    B: DeviceBuffer<T>,
{
    let mut maximum = max_offset(operands.input.layout)?;
    maximum = maximum.max(max_offset(operands.weight.layout)?);
    maximum = maximum.max(max_offset(operands.grad_output.layout)?);
    if let Some(target) = operands.gradients.input {
        maximum = maximum.max(max_offset(target.layout)?);
    }
    if let Some(target) = operands.gradients.weight {
        maximum = maximum.max(max_offset(target.layout)?);
    }
    if let Some(target) = operands.gradients.bias {
        maximum = maximum.max(max_offset(target.layout)?);
    }
    Ok(maximum)
}

fn max_offset<const R: usize>(layout: &Layout<R>) -> Result<usize> {
    layout
        .checked_min_max_offsets()
        .map(|(_, maximum)| maximum)
        .map_err(map_layout_err)
}

pub(super) fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}
