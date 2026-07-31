use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionOps, AttentionPlan,
    AttentionScalar, ComputeDevice, CudaC, DialectScalar, HephaestusError, Result,
    plan_attention_backward, plan_attention_forward,
};
use leto::Layout;

use super::kernel::{
    GradientKernel, backward_preflight_source, backward_source, backward_validation_source,
    forward_preflight_source, forward_source, score_gradient_source,
};
use super::metadata::{BackwardMeta, BackwardPreflightMeta, ForwardMeta};
use super::prepared::{
    PreparedAttentionBackward, PreparedAttentionForward, PreparedBackwardSpec, PreparedForwardSpec,
    PreparedGradientSpec, compile,
};
use super::resources::{
    backward_aliases, forward_aliases, validate_backward_device, validate_forward_device,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

trait CudaAttentionScalar: AttentionScalar + DialectScalar<CudaC> {
    const EXPONENTIAL: &'static str;
    const EPSILON: &'static str;
}

impl CudaAttentionScalar for f32 {
    const EXPONENTIAL: &'static str = "expf";
    const EPSILON: &'static str = "1.1920928955078125e-7f";
}

impl CudaAttentionScalar for f64 {
    const EXPONENTIAL: &'static str = "exp";
    const EPSILON: &'static str = "2.2204460492503130808472633361816e-16";
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
        let preflight_elements = forward_preflight_elements(&operands, rows)?;
        let preflight_kernel = compile::<T>(
            device,
            "attention_forward_preflight",
            forward_preflight_source(T::TYPE_TOKEN, T::EXPONENTIAL),
        )?;
        let kernel = if rows == 0 {
            None
        } else {
            Some(compile::<T>(
                device,
                "attention_forward",
                forward_source(T::TYPE_TOKEN, T::EXPONENTIAL),
            )?)
        };
        let status = device.alloc_uninitialized::<u32>(1)?;
        Ok(PreparedAttentionForward::new(
            device,
            operands,
            PreparedForwardSpec {
                preflight_kernel,
                kernel,
                status,
                metadata,
                preflight_elements,
                rows,
            },
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
        let validation_elements = backward_preflight_elements(&operands, score_rows)?;
        let validation_kernel = compile::<T>(
            device,
            "attention_backward_preflight",
            backward_validation_source(T::TYPE_TOKEN, T::EPSILON),
        )?;
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
        let status = device.alloc_uninitialized::<u32>(1)?;
        let score_metadata = BackwardMeta::new(&operands, operands.query.layout)?;
        let preflight_metadata = BackwardPreflightMeta::new(&operands)?;
        Ok(PreparedAttentionBackward::new(
            device,
            operands,
            PreparedBackwardSpec {
                validation_kernel,
                score_kernel,
                score_gradient,
                status,
                score_metadata,
                preflight_metadata,
                validation_elements,
                score_rows,
                gradients: [query, key, value],
            },
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
            let preflight_kernel = if elements == 0 {
                None
            } else {
                Some(compile::<T>(
                    device,
                    kind.preflight_entry(),
                    backward_preflight_source(T::TYPE_TOKEN, kind),
                )?)
            };
            Ok(PreparedGradientSpec {
                kind,
                preflight_kernel,
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

fn forward_preflight_elements<T>(
    operands: &AttentionForwardOperands<'_, CudaBuffer<T>, T>,
    rows: usize,
) -> Result<usize> {
    let keep = operands
        .mask
        .grouped_keep()
        .map_or(Ok(0), |mask| layout_size(mask.view().layout, "keep mask"))?;
    Ok([
        rows,
        layout_size(operands.query.layout, "query")?,
        layout_size(operands.key.layout, "key")?,
        layout_size(operands.value.layout, "value")?,
        keep,
        1,
    ]
    .into_iter()
    .max()
    .expect("invariant: forward preflight candidate set is nonempty"))
}

fn backward_preflight_elements<T>(
    operands: &AttentionBackwardOperands<'_, CudaBuffer<T>, T>,
    rows: usize,
) -> Result<usize> {
    let query_gradient = operands.gradients.query.map_or(Ok(0), |gradient| {
        layout_size(gradient.layout, "query gradient")
    })?;
    let key_gradient = operands.gradients.key.map_or(Ok(0), |gradient| {
        layout_size(gradient.layout, "key gradient")
    })?;
    let value_gradient = operands.gradients.value.map_or(Ok(0), |gradient| {
        layout_size(gradient.layout, "value gradient")
    })?;
    Ok([
        rows,
        layout_size(operands.grad_output.layout, "output gradient")?,
        layout_size(operands.query.layout, "query")?,
        layout_size(operands.key.layout, "key")?,
        layout_size(operands.value.layout, "value")?,
        layout_size(operands.weights.layout, "weights")?,
        query_gradient,
        key_gradient,
        value_gradient,
        1,
    ]
    .into_iter()
    .max()
    .expect("invariant: backward preflight candidate set is nonempty"))
}

fn layout_size<const R: usize>(layout: &Layout<R>, name: &str) -> Result<usize> {
    layout
        .checked_size()
        .map_err(|error| HephaestusError::InvalidConfiguration {
            message: format!("attention {name} layout size rejected: {error}"),
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
