use super::super::validation::invalid;
use crate::domain::error::Result;

use super::plans::ConvolutionPlan;
use super::plans::TransposedConvolutionPlan;

pub(crate) fn validate_plan_address_limit<const S: usize, const P: usize>(
    max_inclusive: usize,
    scalar_values: [usize; 5],
    spatial_values: [&[usize; S]; P],
) -> Result<()> {
    let exceeds_limit = scalar_values
        .into_iter()
        .chain(
            spatial_values
                .into_iter()
                .flat_map(|values| values.iter().copied()),
        )
        .any(|value| value > max_inclusive);
    if exceeds_limit {
        return Err(invalid(format!(
            "convolution plan exceeds backend address limit {max_inclusive}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_regular_projection_limit<const S: usize>(
    plan: &ConvolutionPlan<S>,
    max_inclusive: usize,
) -> Result<()> {
    for axis in 0..S {
        let forward_projection = checked_projection_sum(
            plan.output_spatial[axis].saturating_sub(1),
            plan.parameters.stride()[axis],
            plan.kernel_spatial[axis].saturating_sub(1),
            plan.parameters.dilation()[axis],
        )?;
        let backward_numerator = plan.input_spatial[axis]
            .saturating_sub(1)
            .checked_add(plan.parameters.padding()[axis])
            .ok_or_else(|| invalid("convolution projection address overflows"))?;
        validate_projection_values(max_inclusive, [forward_projection, backward_numerator])?;
    }
    Ok(())
}

pub(crate) fn validate_transposed_projection_limit<const S: usize>(
    plan: &TransposedConvolutionPlan<S>,
    max_inclusive: usize,
) -> Result<()> {
    for axis in 0..S {
        let direct_projection = checked_projection_sum(
            plan.input_spatial[axis].saturating_sub(1),
            plan.parameters.stride()[axis],
            plan.kernel_spatial[axis].saturating_sub(1),
            plan.parameters.dilation()[axis],
        )?;
        let inverse_numerator = plan.output_spatial[axis]
            .saturating_sub(1)
            .checked_add(plan.parameters.padding()[axis])
            .ok_or_else(|| invalid("transposed convolution projection address overflows"))?;
        validate_projection_values(max_inclusive, [direct_projection, inverse_numerator])?;
    }
    Ok(())
}

fn checked_projection_sum(
    first_extent: usize,
    first_step: usize,
    second_extent: usize,
    second_step: usize,
) -> Result<usize> {
    first_extent
        .checked_mul(first_step)
        .and_then(|first| {
            second_extent
                .checked_mul(second_step)
                .and_then(|second| first.checked_add(second))
        })
        .ok_or_else(|| invalid("convolution projection address overflows"))
}

fn validate_projection_values(max_inclusive: usize, values: [usize; 2]) -> Result<()> {
    if values.into_iter().any(|value| value > max_inclusive) {
        return Err(invalid(format!(
            "convolution projection exceeds backend address limit {max_inclusive}"
        )));
    }
    Ok(())
}
