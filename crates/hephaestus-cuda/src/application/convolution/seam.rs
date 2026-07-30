use bytemuck::Pod;
use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionOps, CudaC, DialectScalar,
    HephaestusError, Result, StridedView, plan_convolution_backward, plan_convolution_forward,
    plan_transposed_convolution_backward, plan_transposed_convolution_forward,
};
use leto::{ConvolutionParameters, TransposedConvolutionParameters};

use super::kernel::{backward_source, forward_source};
use super::metadata::ConvolutionMeta;
use super::prepared::{
    PreparedConvolutionBackward, PreparedConvolutionForward, PreparedConvolutionGradient, compile,
};
use super::resources::{
    backward_aliases, forward_aliases, validate_backward_device, validate_forward_device,
};
use super::routing::{
    BiasMode, ConvolutionDirection, GradientTarget, forward_label, gradient_label,
    validate_spatial_rank,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

trait CudaConvolutionScalar: DialectScalar<CudaC> + Pod + Send + Sync + 'static {}

impl CudaConvolutionScalar for f32 {}
impl CudaConvolutionScalar for f64 {}

/// CUDA implementation of provider-owned convolution.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaConvolutionOps;

impl<T> ConvolutionOps<CudaDevice, T> for CudaConvolutionOps
where
    T: CudaConvolutionScalar,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = PreparedConvolutionForward<'a, T>
    where
        CudaDevice: 'a,
        T: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = PreparedConvolutionBackward<'a, T>
    where
        CudaDevice: 'a,
        T: 'a;
    type PreparedTransposedForward<'a, const R: usize, const S: usize>
        = PreparedConvolutionForward<'a, T>
    where
        CudaDevice: 'a,
        T: 'a;
    type PreparedTransposedBackward<'a, const R: usize, const S: usize>
        = PreparedConvolutionBackward<'a, T>
    where
        CudaDevice: 'a,
        T: 'a;

    fn prepare_convolution_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a CudaDevice,
        operands: ConvolutionForwardOperands<'a, CudaBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedForward<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        validate_forward_device(device, &operands)?;
        let plan = plan_convolution_forward::<T, _, R, S>(
            &operands,
            parameters,
            forward_aliases(&operands),
        )?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::regular_forward(
            operands.input.layout,
            operands.weight.layout,
            operands.output.layout,
            operands.bias.map(|bias| bias.layout),
            parameters,
        )?;
        prepare_forward(
            device,
            operands,
            metadata,
            plan.output_elements,
            ConvolutionDirection::Regular,
        )
    }

    fn dispatch_convolution_forward<const R: usize, const S: usize>(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_convolution_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a CudaDevice,
        operands: ConvolutionBackwardOperands<'a, CudaBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedBackward<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        validate_backward_device(device, &operands)?;
        let plan = plan_convolution_backward::<T, _, R, S>(
            &operands,
            parameters,
            backward_aliases(&operands),
        )?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::regular_backward(
            operands.input.layout,
            operands.weight.layout,
            operands.grad_output.layout,
            parameters,
        )?;
        prepare_backward(device, operands, metadata, ConvolutionDirection::Regular)
    }

    fn dispatch_convolution_backward<const R: usize, const S: usize>(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_convolution_transposed_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a CudaDevice,
        operands: ConvolutionForwardOperands<'a, CudaBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedForward<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        validate_forward_device(device, &operands)?;
        let plan = plan_transposed_convolution_forward::<T, _, R, S>(
            &operands,
            parameters,
            forward_aliases(&operands),
        )?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::transposed_forward(
            operands.input.layout,
            operands.weight.layout,
            operands.output.layout,
            operands.bias.map(|bias| bias.layout),
            parameters,
        )?;
        prepare_forward(
            device,
            operands,
            metadata,
            plan.output_elements,
            ConvolutionDirection::Transposed,
        )
    }

    fn dispatch_convolution_transposed_forward<const R: usize, const S: usize>(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedTransposedForward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_convolution_transposed_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a CudaDevice,
        operands: ConvolutionBackwardOperands<'a, CudaBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedBackward<'a, R, S>> {
        validate_spatial_rank::<S>()?;
        validate_backward_device(device, &operands)?;
        let plan = plan_transposed_convolution_backward::<T, _, R, S>(
            &operands,
            parameters,
            backward_aliases(&operands),
        )?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::transposed_backward(
            operands.input.layout,
            operands.weight.layout,
            operands.grad_output.layout,
            parameters,
        )?;
        prepare_backward(device, operands, metadata, ConvolutionDirection::Transposed)
    }

    fn dispatch_convolution_transposed_backward<const R: usize, const S: usize>(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedTransposedBackward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn prepare_forward<'a, T, const R: usize>(
    device: &'a CudaDevice,
    operands: ConvolutionForwardOperands<'a, CudaBuffer<T>, R>,
    metadata: ConvolutionMeta,
    elements: usize,
    direction: ConvolutionDirection,
) -> Result<PreparedConvolutionForward<'a, T>>
where
    T: CudaConvolutionScalar,
{
    let (_, entry) = forward_label(direction);
    let bias_mode = if operands.bias.is_some() {
        BiasMode::Present
    } else {
        BiasMode::Absent
    };
    let kernel = if elements == 0 {
        None
    } else {
        let source = forward_source(T::TYPE_TOKEN, entry, direction, bias_mode, R - 2);
        Some(compile::<T>(
            device,
            entry,
            R - 2,
            bias_mode == BiasMode::Present,
            source,
        )?)
    };
    Ok(PreparedConvolutionForward::new(
        device, kernel, operands, metadata, elements,
    ))
}

fn prepare_backward<'a, T, const R: usize>(
    device: &'a CudaDevice,
    operands: ConvolutionBackwardOperands<'a, CudaBuffer<T>, R>,
    metadata: ConvolutionMeta,
    direction: ConvolutionDirection,
) -> Result<PreparedConvolutionBackward<'a, T>>
where
    T: CudaConvolutionScalar,
{
    let input = operands
        .gradients
        .input
        .map(|target| {
            prepare_gradient(
                device,
                &operands,
                target,
                metadata,
                direction,
                GradientTarget::Input,
            )
        })
        .transpose()?;
    let weight = operands
        .gradients
        .weight
        .map(|target| {
            prepare_gradient(
                device,
                &operands,
                target,
                metadata,
                direction,
                GradientTarget::Weight,
            )
        })
        .transpose()?;
    let bias = operands
        .gradients
        .bias
        .map(|target| prepare_bias_gradient(device, &operands, target, metadata, direction))
        .transpose()?;
    Ok(PreparedConvolutionBackward::new(
        device, input, weight, bias,
    ))
}

fn prepare_gradient<'a, T, const R: usize>(
    device: &CudaDevice,
    operands: &ConvolutionBackwardOperands<'a, CudaBuffer<T>, R>,
    target: StridedView<'a, CudaBuffer<T>, R>,
    metadata: ConvolutionMeta,
    direction: ConvolutionDirection,
    gradient: GradientTarget,
) -> Result<PreparedConvolutionGradient<'a, T>>
where
    T: CudaConvolutionScalar,
{
    let (_, entry) = gradient_label(direction, gradient);
    let elements = target.layout.checked_size().map_err(layout_error)?;
    let kernel = if elements == 0 {
        None
    } else {
        let source = backward_source(T::TYPE_TOKEN, entry, direction, gradient, R - 2);
        Some(compile::<T>(device, entry, R - 2, false, source)?)
    };
    let metadata = metadata.with_target(target.layout)?;
    let second = match gradient {
        GradientTarget::Input => operands.weight.buffer,
        GradientTarget::Weight => operands.input.buffer,
        GradientTarget::Bias => {
            return Err(HephaestusError::InvalidConfiguration {
                message: "bias gradient must use the bias preparation path".to_string(),
            });
        }
    };
    Ok(PreparedConvolutionBackward::gradient(
        kernel,
        operands.grad_output.buffer,
        Some(second),
        target.buffer,
        metadata,
        elements,
    ))
}

fn prepare_bias_gradient<'a, T, const R: usize>(
    device: &CudaDevice,
    operands: &ConvolutionBackwardOperands<'a, CudaBuffer<T>, R>,
    target: StridedView<'a, CudaBuffer<T>, 1>,
    metadata: ConvolutionMeta,
    direction: ConvolutionDirection,
) -> Result<PreparedConvolutionGradient<'a, T>>
where
    T: CudaConvolutionScalar,
{
    let gradient = GradientTarget::Bias;
    let (_, entry) = gradient_label(direction, gradient);
    let elements = target.layout.checked_size().map_err(layout_error)?;
    let kernel = if elements == 0 {
        None
    } else {
        let source = backward_source(T::TYPE_TOKEN, entry, direction, gradient, R - 2);
        Some(compile::<T>(device, entry, R - 2, false, source)?)
    };
    Ok(PreparedConvolutionBackward::gradient(
        kernel,
        operands.grad_output.buffer,
        None,
        target.buffer,
        metadata.with_target(target.layout)?,
        elements,
    ))
}

fn signed_address_limit() -> usize {
    usize::try_from(i32::MAX).expect("invariant: supported CUDA hosts represent i32 in usize")
}

fn layout_error(error: leto::LetoError) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: format!("convolution target layout rejected: {error}"),
    }
}
