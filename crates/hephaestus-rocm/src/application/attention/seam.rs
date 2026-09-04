use eunomia::Pod;
use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionOps, AttentionScalar,
    ComputeDevice, DialectScalar, HephaestusError, HipC, Result, StridedView,
    plan_attention_backward, plan_attention_forward,
};
use leto::Layout;

use super::kernel::{
    CANDIDATE_ENTRY, FORWARD_ENTRY, FORWARD_PREFLIGHT_ENTRY, FiniteOperand, GradientTarget,
    PROBABILITY_ENTRY, SCORE_ENTRY, backward_source, candidate_source, finite_source,
    forward_arithmetic_source, forward_source, gradient_preflight_source, probability_source,
    score_source,
};
use super::metadata::{AttentionMeta, validate_plan};
use super::prepared::{
    PreparedAttentionBackward, PreparedAttentionForward, PreparedAttentionGradient, PreparedFinite,
    compile,
};
use super::resources::{
    backward_aliases, forward_aliases, validate_backward_device, validate_forward_device,
};
use crate::{RocmBuffer, RocmDevice};

trait HipAttentionScalar: AttentionScalar + DialectScalar<HipC> + Pod + Send + Sync + 'static {}

impl HipAttentionScalar for f32 {}
impl HipAttentionScalar for f64 {}

/// ROCm implementation of provider-owned scaled dot-product attention.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmAttentionOps;

impl<T> AttentionOps<RocmDevice, T> for RocmAttentionOps
where
    T: HipAttentionScalar,
{
    type PreparedForward<'a>
        = PreparedAttentionForward<'a, T>
    where
        RocmDevice: 'a,
        T: 'a;
    type PreparedBackward<'a>
        = PreparedAttentionBackward<'a, T>
    where
        RocmDevice: 'a,
        T: 'a;

    fn prepare_attention_forward<'a>(
        &self,
        device: &'a RocmDevice,
        operands: AttentionForwardOperands<'a, RocmBuffer<T>, T>,
    ) -> Result<Self::PreparedForward<'a>> {
        validate_forward_device(device, &operands)?;
        let plan = plan_attention_forward(&operands, forward_aliases(&operands))?;
        validate_plan(plan)?;
        let metadata = AttentionMeta::forward(&operands)?;
        let rows = checked_product(plan.batch, plan.query_sequence, "forward row count")?;
        let key_rows = checked_product(plan.batch, plan.key_sequence, "key row count")?;
        let keep = operands
            .mask
            .grouped_keep()
            .map_or(operands.query.buffer, |mask| mask.view().buffer);
        let keep_elements = operands
            .mask
            .grouped_keep()
            .map(|mask| logical_elements(mask.view().layout, "keep mask"))
            .transpose()?
            .unwrap_or(0);
        let finite = [
            prepare_finite(
                device,
                operands.query.buffer,
                metadata,
                checked_product(rows, plan.key_feature, "query element count")?,
                FiniteOperand::Query,
            )?,
            prepare_finite(
                device,
                operands.key.buffer,
                metadata,
                checked_product(key_rows, plan.key_feature, "key element count")?,
                FiniteOperand::Key,
            )?,
            prepare_finite(
                device,
                operands.value.buffer,
                metadata,
                checked_product(key_rows, plan.value_feature, "value element count")?,
                FiniteOperand::Value,
            )?,
            prepare_finite(device, keep, metadata, keep_elements, FiniteOperand::Keep)?,
        ];
        let arithmetic_kernel =
            compile_optional::<T>(device, FORWARD_PREFLIGHT_ENTRY, rows, || {
                forward_arithmetic_source(T::TYPE_TOKEN)
            })?;
        let kernel = compile_optional::<T>(device, FORWARD_ENTRY, rows, || {
            forward_source(T::TYPE_TOKEN)
        })?;
        let status = device.alloc_zeroed::<u32>(1)?;
        Ok(PreparedAttentionForward::new(
            device,
            finite,
            arithmetic_kernel,
            kernel,
            status,
            operands.query.buffer,
            operands.key.buffer,
            operands.value.buffer,
            keep,
            operands.output.buffer,
            operands.weights.buffer,
            operands.scale,
            metadata,
            rows,
        ))
    }

    fn dispatch_attention_forward(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_attention_backward<'a>(
        &self,
        device: &'a RocmDevice,
        operands: AttentionBackwardOperands<'a, RocmBuffer<T>, T>,
    ) -> Result<Self::PreparedBackward<'a>> {
        validate_backward_device(device, &operands)?;
        let plan = plan_attention_backward(&operands, backward_aliases(&operands))?;
        validate_plan(plan)?;
        let metadata = AttentionMeta::backward(&operands)?;
        let rows = checked_product(plan.batch, plan.query_sequence, "backward row count")?;
        let key_rows = checked_product(plan.batch, plan.key_sequence, "backward key row count")?;
        let finite = [
            prepare_finite(
                device,
                operands.grad_output.buffer,
                metadata,
                checked_product(rows, plan.value_feature, "output-gradient element count")?,
                FiniteOperand::GradOutput,
            )?,
            prepare_finite(
                device,
                operands.query.buffer,
                metadata,
                checked_product(rows, plan.key_feature, "query element count")?,
                FiniteOperand::Query,
            )?,
            prepare_finite(
                device,
                operands.key.buffer,
                metadata,
                checked_product(key_rows, plan.key_feature, "key element count")?,
                FiniteOperand::Key,
            )?,
            prepare_finite(
                device,
                operands.value.buffer,
                metadata,
                checked_product(key_rows, plan.value_feature, "value element count")?,
                FiniteOperand::Value,
            )?,
            prepare_finite(
                device,
                operands.weights.buffer,
                metadata,
                plan.score_elements,
                FiniteOperand::Weights,
            )?,
        ];
        let probability_kernel = compile_optional::<T>(device, PROBABILITY_ENTRY, rows, || {
            probability_source(T::TYPE_TOKEN)
        })?;
        let needs_score = operands.gradients.query.is_some() || operands.gradients.key.is_some();
        let candidate = needs_score
            .then(|| device.alloc_uninitialized::<T>(plan.score_elements))
            .transpose()?;
        let score_gradient = needs_score
            .then(|| device.alloc_uninitialized::<T>(plan.score_elements))
            .transpose()?;
        let candidate_kernel = if needs_score {
            compile_optional::<T>(device, CANDIDATE_ENTRY, plan.score_elements, || {
                candidate_source(T::TYPE_TOKEN)
            })?
        } else {
            None
        };
        let score_kernel = if needs_score {
            compile_optional::<T>(device, SCORE_ENTRY, rows, || score_source(T::TYPE_TOKEN))?
        } else {
            None
        };
        let gradients = [
            operands
                .gradients
                .query
                .map(|destination| {
                    prepare_gradient(
                        device,
                        &operands,
                        destination,
                        metadata,
                        GradientTarget::Query,
                    )
                })
                .transpose()?,
            operands
                .gradients
                .key
                .map(|destination| {
                    prepare_gradient(
                        device,
                        &operands,
                        destination,
                        metadata,
                        GradientTarget::Key,
                    )
                })
                .transpose()?,
            operands
                .gradients
                .value
                .map(|destination| {
                    prepare_gradient(
                        device,
                        &operands,
                        destination,
                        metadata,
                        GradientTarget::Value,
                    )
                })
                .transpose()?,
        ];
        let status = device.alloc_zeroed::<u32>(1)?;
        Ok(PreparedAttentionBackward::new(
            device,
            finite,
            probability_kernel,
            candidate_kernel,
            score_kernel,
            status,
            candidate,
            score_gradient,
            &operands,
            metadata,
            rows,
            plan.score_elements,
            gradients,
        ))
    }

    fn dispatch_attention_backward(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn prepare_finite<'a, T>(
    device: &RocmDevice,
    source: &'a RocmBuffer<T>,
    metadata: AttentionMeta,
    elements: usize,
    operand: FiniteOperand,
) -> Result<PreparedFinite<'a, T>>
where
    T: HipAttentionScalar,
{
    let kernel = compile_optional::<T>(device, operand.entry(), elements, || {
        finite_source(T::TYPE_TOKEN, operand)
    })?;
    Ok(PreparedFinite::new(kernel, source, metadata, elements))
}

fn prepare_gradient<'a, T>(
    device: &RocmDevice,
    operands: &AttentionBackwardOperands<'a, RocmBuffer<T>, T>,
    destination: StridedView<'a, RocmBuffer<T>, 3>,
    metadata: AttentionMeta,
    target: GradientTarget,
) -> Result<PreparedAttentionGradient<'a, T>>
where
    T: HipAttentionScalar,
{
    let elements = logical_elements(destination.layout, "gradient")?;
    let preflight_kernel =
        compile_optional::<T>(device, target.preflight_entry(), elements, || {
            gradient_preflight_source(T::TYPE_TOKEN, target)
        })?;
    let kernel = compile_optional::<T>(device, target.entry(), elements, || {
        backward_source(T::TYPE_TOKEN, target)
    })?;
    let source = match target {
        GradientTarget::Query => operands.key.buffer,
        GradientTarget::Key | GradientTarget::Value => operands.query.buffer,
    };
    Ok(PreparedAttentionGradient::new(
        target,
        preflight_kernel,
        kernel,
        source,
        destination.buffer,
        metadata.with_destination(destination.layout)?,
        elements,
    ))
}

fn compile_optional<T: 'static>(
    device: &RocmDevice,
    entry: &'static str,
    elements: usize,
    source: impl FnOnce() -> String,
) -> Result<Option<std::sync::Arc<crate::application::pipeline::RocmKernel>>> {
    if elements == 0 {
        Ok(None)
    } else {
        compile::<T>(device, entry, source()).map(Some)
    }
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        invalid(format!(
            "attention {name} overflows: {left} multiplied by {right}"
        ))
    })
}

fn logical_elements<const R: usize>(layout: &Layout<R>, name: &str) -> Result<usize> {
    layout
        .checked_size()
        .map_err(|error| invalid(format!("attention {name} layout rejected: {error}")))
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_provider<T>()
    where
        T: AttentionScalar,
        RocmAttentionOps: AttentionOps<RocmDevice, T>,
    {
    }

    #[test]
    fn provider_is_monomorphized_for_native_attention_scalars() {
        assert_provider::<f32>();
        assert_provider::<f64>();
    }
}
