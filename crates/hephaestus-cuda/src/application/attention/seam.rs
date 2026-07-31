use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionOps, AttentionPlan,
    AttentionScalar, ComputeDevice, CudaC, DialectScalar, HephaestusError, Result,
    plan_attention_backward, plan_attention_forward,
};

use super::kernel::{GradientKernel, backward_source, forward_source, score_gradient_source};
use super::metadata::{BackwardMeta, ForwardMeta};
use super::prepared::{
    PreparedAttentionBackward, PreparedAttentionForward, PreparedGradientSpec, compile,
};
use super::resources::{
    backward_aliases, forward_aliases, validate_backward_device, validate_forward_device,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

trait CudaAttentionScalar: AttentionScalar + DialectScalar<CudaC> {
    const EXPONENTIAL: &'static str;
}

impl CudaAttentionScalar for f32 {
    const EXPONENTIAL: &'static str = "expf";
}

impl CudaAttentionScalar for f64 {
    const EXPONENTIAL: &'static str = "exp";
}

/// CUDA implementation of provider-owned scaled dot-product attention.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaAttentionOps;

impl<T> AttentionOps<CudaDevice, T> for CudaAttentionOps
where
    T: CudaAttentionScalar,
{
    type PreparedForward<'a>
        = PreparedAttentionForward<'a, T>
    where
        CudaDevice: 'a,
        T: 'a;
    type PreparedBackward<'a>
        = PreparedAttentionBackward<'a, T>
    where
        CudaDevice: 'a,
        T: 'a;

    fn prepare_attention_forward<'a>(
        &self,
        device: &'a CudaDevice,
        operands: AttentionForwardOperands<'a, CudaBuffer<T>, T>,
    ) -> Result<Self::PreparedForward<'a>> {
        validate_forward_device(device, &operands)?;
        let plan = plan_attention_forward(&operands, forward_aliases(&operands))?;
        plan.validate_address_limit(cuda_address_limit())?;
        let metadata = ForwardMeta::new(&operands)?;
        let rows = score_rows(plan)?;
        let kernel = if rows == 0 {
            None
        } else {
            Some(compile::<T>(
                device,
                "attention_forward",
                forward_source(T::TYPE_TOKEN, T::EXPONENTIAL),
            )?)
        };
        Ok(PreparedAttentionForward::new(
            device, kernel, operands, metadata, rows,
        ))
    }

    fn dispatch_attention_forward(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_attention_backward<'a>(
        &self,
        device: &'a CudaDevice,
        operands: AttentionBackwardOperands<'a, CudaBuffer<T>, T>,
    ) -> Result<Self::PreparedBackward<'a>> {
        validate_backward_device(device, &operands)?;
        let plan = plan_attention_backward(&operands, backward_aliases(&operands))?;
        plan.validate_address_limit(cuda_address_limit())?;
        let score_rows = score_rows(plan)?;
        let query = prepare_gradient(device, &operands, GradientKernel::Query)?;
        let key = prepare_gradient(device, &operands, GradientKernel::Key)?;
        let value = prepare_gradient(device, &operands, GradientKernel::Value)?;
        let needs_score = (query.is_some() || key.is_some()) && plan.score_elements != 0;
        let score_kernel = if needs_score {
            Some(compile::<T>(
                device,
                "attention_score_gradient",
                score_gradient_source(T::TYPE_TOKEN),
            )?)
        } else {
            None
        };
        let score_gradient =
            device.alloc_uninitialized::<T>(if needs_score { plan.score_elements } else { 0 })?;
        let score_metadata = BackwardMeta::new(&operands, operands.query.layout)?;
        Ok(PreparedAttentionBackward::new(
            device,
            score_kernel,
            score_gradient,
            operands,
            score_metadata,
            score_rows,
            [query, key, value],
        ))
    }

    fn dispatch_attention_backward(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn prepare_gradient<'a, T>(
    device: &CudaDevice,
    operands: &AttentionBackwardOperands<'a, CudaBuffer<T>, T>,
    kind: GradientKernel,
) -> Result<Option<PreparedGradientSpec<'a, T>>>
where
    T: CudaAttentionScalar,
{
    let target = match kind {
        GradientKernel::Query => operands.gradients.query,
        GradientKernel::Key => operands.gradients.key,
        GradientKernel::Value => operands.gradients.value,
    };
    target
        .map(|target| {
            let elements = target.layout.checked_size().map_err(layout_error)?;
            let kernel = if elements == 0 {
                None
            } else {
                Some(compile::<T>(
                    device,
                    kind.entry(),
                    backward_source(T::TYPE_TOKEN, kind),
                )?)
            };
            Ok(PreparedGradientSpec {
                kind,
                kernel,
                target: target.buffer,
                metadata: BackwardMeta::new(operands, target.layout)?,
                elements,
            })
        })
        .transpose()
}

fn score_rows(plan: AttentionPlan) -> Result<usize> {
    plan.batch.checked_mul(plan.query_sequence).ok_or_else(|| {
        HephaestusError::InvalidConfiguration {
            message: "attention row count overflows usize".to_string(),
        }
    })
}

fn cuda_address_limit() -> usize {
    usize::try_from(i64::MAX).unwrap_or(usize::MAX)
}

fn layout_error(error: leto::LetoError) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: format!("attention target layout rejected: {error}"),
    }
}
