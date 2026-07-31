use bytemuck::Pod;
use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionOps, AttentionScalar,
    DialectScalar, HephaestusError, HipC, Result, StridedView, plan_attention_backward,
    plan_attention_forward,
};

use super::kernel::{FORWARD_ENTRY, GradientTarget, backward_source, forward_source};
use super::metadata::{AttentionMeta, validate_plan};
use super::prepared::{PreparedAttentionBackward, PreparedAttentionForward, compile};
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
        let rows = plan
            .batch
            .checked_mul(plan.query_sequence)
            .ok_or_else(|| invalid("attention forward row count overflows"))?;
        let kernel = if rows == 0 {
            None
        } else {
            Some(compile::<T>(
                device,
                FORWARD_ENTRY,
                forward_source(T::TYPE_TOKEN),
            )?)
        };
        let keep = operands
            .mask
            .grouped_keep()
            .map_or(operands.query.buffer, |mask| mask.view().buffer);
        Ok(PreparedAttentionForward::new(
            device,
            kernel,
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

        let query = operands
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
            .transpose()?;
        let key = operands
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
            .transpose()?;
        let value = operands
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
            .transpose()?;

        Ok(PreparedAttentionBackward::new(device, query, key, value))
    }

    fn dispatch_attention_backward(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn prepare_gradient<'a, T>(
    device: &RocmDevice,
    operands: &AttentionBackwardOperands<'a, RocmBuffer<T>, T>,
    destination: StridedView<'a, RocmBuffer<T>, 3>,
    metadata: AttentionMeta,
    target: GradientTarget,
) -> Result<super::prepared::PreparedAttentionGradient<'a, T>>
where
    T: HipAttentionScalar,
{
    let elements = destination.layout.checked_size().map_err(|error| {
        invalid(format!(
            "attention gradient layout rejected before ROCm preparation: {error}"
        ))
    })?;
    let kernel = if elements == 0 {
        None
    } else {
        Some(compile::<T>(
            device,
            target.entry(),
            backward_source(T::TYPE_TOKEN, target),
        )?)
    };
    Ok(PreparedAttentionBackward::gradient(
        kernel,
        operands.grad_output.buffer,
        operands.query.buffer,
        operands.key.buffer,
        operands.value.buffer,
        operands.weights.buffer,
        destination.buffer,
        operands.scale,
        metadata.with_destination(destination.layout)?,
        elements,
    ))
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
