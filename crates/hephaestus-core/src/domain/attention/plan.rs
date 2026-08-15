use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;

use super::validation::{expect_shape, invalid, max_offset, validate_readonly, validate_writable};
use super::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionGradientViews, AttentionScalar,
};

/// Validated scaled dot-product attention dimensions and launch bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionPlan {
    /// Execution batch extent, including flattened heads.
    pub batch: usize,
    /// Query sequence extent.
    pub query_sequence: usize,
    /// Key/value sequence extent.
    pub key_sequence: usize,
    /// Query/key feature extent.
    pub key_feature: usize,
    /// Value/output feature extent.
    pub value_feature: usize,
    /// Logical post-softmax score count.
    pub score_elements: usize,
    /// Largest physical element offset touched by any operand.
    pub max_physical_offset: usize,
}

impl AttentionPlan {
    /// Validate every dimension and offset narrowed by a backend kernel.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error when a value exceeds the inclusive
    /// backend address limit.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        if [
            self.batch,
            self.query_sequence,
            self.key_sequence,
            self.key_feature,
            self.value_feature,
            self.score_elements,
            self.max_physical_offset,
        ]
        .into_iter()
        .any(|value| value > max_inclusive)
        {
            return Err(invalid(format!(
                "attention plan exceeds backend address limit {max_inclusive}"
            )));
        }
        Ok(())
    }
}

/// Validate attention forward before backend preparation.
///
/// `illegal_aliasing` is the backend's aggregate buffer-identity result for
/// writable outputs against readable operands and each other.
///
/// # Errors
///
/// Returns a typed shape, storage, layout, scalar, overflow, or alias error.
pub fn plan_attention_forward<T, B>(
    operands: &AttentionForwardOperands<'_, B, T>,
    illegal_aliasing: bool,
) -> Result<AttentionPlan>
where
    T: AttentionScalar,
    B: DeviceBuffer<T>,
{
    validate_readonly::<T, _, 3>(operands.query.buffer, operands.query.layout)?;
    validate_readonly::<T, _, 3>(operands.key.buffer, operands.key.layout)?;
    validate_readonly::<T, _, 3>(operands.value.buffer, operands.value.layout)?;
    validate_writable::<T, _>(operands.output.buffer, operands.output.layout, "output")?;
    validate_writable::<T, _>(operands.weights.buffer, operands.weights.layout, "weights")?;
    reject_aliasing(illegal_aliasing)?;

    let plan = dimensions(
        operands.query.layout.shape(),
        operands.key.layout.shape(),
        operands.value.layout.shape(),
    )?;
    expect_shape(
        "output",
        operands.output.layout.shape(),
        [plan.batch, plan.query_sequence, plan.value_feature],
    )?;
    expect_shape(
        "weights",
        operands.weights.layout.shape(),
        [plan.batch, plan.query_sequence, plan.key_sequence],
    )?;
    validate_scale(operands.scale)?;
    validate_mask::<T, _>(&operands.mask, plan)?;

    Ok(AttentionPlan {
        max_physical_offset: forward_max_offset(operands)?,
        ..plan
    })
}

/// Validate additive attention backward before backend preparation.
///
/// `illegal_aliasing` is the backend's aggregate buffer-identity result for
/// every selected destination against readable operands and other targets.
///
/// # Errors
///
/// Returns a typed empty-target, shape, storage, layout, scalar, overflow, or
/// alias error.
pub fn plan_attention_backward<T, B>(
    operands: &AttentionBackwardOperands<'_, B, T>,
    illegal_aliasing: bool,
) -> Result<AttentionPlan>
where
    T: AttentionScalar,
    B: DeviceBuffer<T>,
{
    if operands.gradients.is_empty() {
        return Err(invalid(
            "attention backward requires at least one gradient destination",
        ));
    }
    validate_readonly::<T, _, 3>(operands.grad_output.buffer, operands.grad_output.layout)?;
    validate_readonly::<T, _, 3>(operands.query.buffer, operands.query.layout)?;
    validate_readonly::<T, _, 3>(operands.key.buffer, operands.key.layout)?;
    validate_readonly::<T, _, 3>(operands.value.buffer, operands.value.layout)?;
    validate_readonly::<T, _, 3>(operands.weights.buffer, operands.weights.layout)?;
    reject_aliasing(illegal_aliasing)?;

    let plan = dimensions(
        operands.query.layout.shape(),
        operands.key.layout.shape(),
        operands.value.layout.shape(),
    )?;
    expect_shape(
        "output gradient",
        operands.grad_output.layout.shape(),
        [plan.batch, plan.query_sequence, plan.value_feature],
    )?;
    expect_shape(
        "weights",
        operands.weights.layout.shape(),
        [plan.batch, plan.query_sequence, plan.key_sequence],
    )?;
    validate_gradients::<T, _>(&operands.gradients, operands, plan)?;
    validate_scale(operands.scale)?;

    Ok(AttentionPlan {
        max_physical_offset: backward_max_offset(operands)?,
        ..plan
    })
}

fn dimensions(query: [usize; 3], key: [usize; 3], value: [usize; 3]) -> Result<AttentionPlan> {
    let [batch, query_sequence, key_feature] = query;
    let [key_batch, key_sequence, key_key_feature] = key;
    let [value_batch, value_sequence, value_feature] = value;
    if key_batch != batch
        || value_batch != batch
        || key_key_feature != key_feature
        || value_sequence != key_sequence
    {
        return Err(invalid(format!(
            "attention input shapes are incompatible: query {query:?}, key {key:?}, value {value:?}"
        )));
    }
    if key_sequence == 0 {
        return Err(invalid(
            "attention key sequence must be nonempty because softmax requires support",
        ));
    }
    let score_elements = batch
        .checked_mul(query_sequence)
        .and_then(|count| count.checked_mul(key_sequence))
        .ok_or_else(|| invalid("attention score element count overflows"))?;
    Ok(AttentionPlan {
        batch,
        query_sequence,
        key_sequence,
        key_feature,
        value_feature,
        score_elements,
        max_physical_offset: 0,
    })
}

fn validate_mask<T, B>(mask: &super::AttentionMask<'_, B>, plan: AttentionPlan) -> Result<()>
where
    B: DeviceBuffer<T>,
{
    let Some(keep) = mask.grouped_keep() else {
        return Ok(());
    };
    let view = keep.view();
    validate_readonly::<T, _, 2>(view.buffer, view.layout)?;
    let [mask_batch, mask_key_sequence] = view.layout.shape();
    if mask_key_sequence != plan.key_sequence {
        return Err(invalid(format!(
            "attention keep-mask key extent {mask_key_sequence} must equal {}",
            plan.key_sequence
        )));
    }
    let represented_batch = mask_batch
        .checked_mul(keep.heads_per_batch().get())
        .ok_or_else(|| invalid("attention keep-mask grouping overflows"))?;
    if represented_batch != plan.batch {
        return Err(invalid(format!(
            "attention keep-mask represents batch {represented_batch}, expected {}",
            plan.batch
        )));
    }
    Ok(())
}

fn validate_gradients<T, B>(
    gradients: &AttentionGradientViews<'_, B>,
    operands: &AttentionBackwardOperands<'_, B, T>,
    plan: AttentionPlan,
) -> Result<()>
where
    T: AttentionScalar,
    B: DeviceBuffer<T>,
{
    if let Some(query) = gradients.query {
        validate_writable::<T, _>(query.buffer, query.layout, "query gradient")?;
        expect_shape(
            "query gradient",
            query.layout.shape(),
            operands.query.layout.shape(),
        )?;
    }
    if let Some(key) = gradients.key {
        validate_writable::<T, _>(key.buffer, key.layout, "key gradient")?;
        expect_shape(
            "key gradient",
            key.layout.shape(),
            operands.key.layout.shape(),
        )?;
    }
    if let Some(value) = gradients.value {
        validate_writable::<T, _>(value.buffer, value.layout, "value gradient")?;
        expect_shape(
            "value gradient",
            value.layout.shape(),
            [plan.batch, plan.key_sequence, plan.value_feature],
        )?;
    }
    Ok(())
}

fn validate_scale<T: AttentionScalar>(scale: T) -> Result<()> {
    if !scale.is_finite() {
        return Err(invalid("attention scale must be finite"));
    }
    Ok(())
}

fn reject_aliasing(illegal_aliasing: bool) -> Result<()> {
    if illegal_aliasing {
        return Err(invalid(
            "attention writable buffers must not alias readable operands or each other",
        ));
    }
    Ok(())
}

fn forward_max_offset<T, B>(operands: &AttentionForwardOperands<'_, B, T>) -> Result<usize> {
    let mut maximum = max_offset(operands.query.layout)?;
    maximum = maximum.max(max_offset(operands.key.layout)?);
    maximum = maximum.max(max_offset(operands.value.layout)?);
    maximum = maximum.max(max_offset(operands.output.layout)?);
    maximum = maximum.max(max_offset(operands.weights.layout)?);
    if let Some(keep) = operands.mask.grouped_keep() {
        maximum = maximum.max(max_offset(keep.view().layout)?);
    }
    Ok(maximum)
}

fn backward_max_offset<T: AttentionScalar, B>(
    operands: &AttentionBackwardOperands<'_, B, T>,
) -> Result<usize> {
    let mut maximum = max_offset(operands.grad_output.layout)?;
    maximum = maximum.max(max_offset(operands.query.layout)?);
    maximum = maximum.max(max_offset(operands.key.layout)?);
    maximum = maximum.max(max_offset(operands.value.layout)?);
    maximum = maximum.max(max_offset(operands.weights.layout)?);
    if let Some(query) = operands.gradients.query {
        maximum = maximum.max(max_offset(query.layout)?);
    }
    if let Some(key) = operands.gradients.key {
        maximum = maximum.max(max_offset(key.layout)?);
    }
    if let Some(value) = operands.gradients.value {
        maximum = maximum.max(max_offset(value.layout)?);
    }
    Ok(maximum)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
