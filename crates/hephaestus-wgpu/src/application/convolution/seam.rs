use core::any::TypeId;

use bytemuck::Pod;
use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionOps, DeviceFeature,
    DialectScalar, HephaestusError, Result, Wgsl, plan_convolution_backward,
    plan_convolution_forward, plan_transposed_convolution_backward,
    plan_transposed_convolution_forward,
};
use leto::{ConvolutionParameters, TransposedConvolutionParameters};

use super::metadata::ConvolutionMeta;
use super::prepared::{PreparedConvolutionBackward, PreparedConvolutionKernel};
use super::resources::{
    backward_aliases, binding, forward_aliases, layout_error, metadata_buffer, raw_binding,
};
use super::routing::{
    forward_pipeline_key, gradient_label, gradient_pipeline_key, validate_spatial_rank,
};
use super::shader::{
    BiasMode, ConvolutionDirection, GradientTarget, backward_shader, forward_shader,
};
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::application::prepared::checked_bind_group;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP_WIDTH: hephaestus_core::BlockWidth = hephaestus_core::BlockWidth::DEFAULT;

trait WgpuConvolutionScalar: DialectScalar<Wgsl> + Pod + Send + Sync + 'static {
    fn validate_capability(device: &WgpuDevice) -> Result<()>;
}

impl WgpuConvolutionScalar for f32 {
    fn validate_capability(_device: &WgpuDevice) -> Result<()> {
        Ok(())
    }
}

impl WgpuConvolutionScalar for f64 {
    fn validate_capability(device: &WgpuDevice) -> Result<()> {
        if device.supports_device_feature(DeviceFeature::ShaderF64) {
            Ok(())
        } else {
            Err(HephaestusError::InvalidConfiguration {
                message: "WGPU convolution requires the ShaderF64 device feature for f64"
                    .to_string(),
            })
        }
    }
}

/// WGPU implementation of provider-owned convolution.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuConvolutionOps;

impl<T> ConvolutionOps<WgpuDevice, T> for WgpuConvolutionOps
where
    T: WgpuConvolutionScalar,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = PreparedConvolutionKernel
    where
        WgpuDevice: 'a,
        T: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = PreparedConvolutionBackward
    where
        WgpuDevice: 'a,
        T: 'a;
    type PreparedTransposedForward<'a, const R: usize, const S: usize>
        = PreparedConvolutionKernel
    where
        WgpuDevice: 'a,
        T: 'a;
    type PreparedTransposedBackward<'a, const R: usize, const S: usize>
        = PreparedConvolutionBackward
    where
        WgpuDevice: 'a,
        T: 'a;

    fn prepare_convolution_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: ConvolutionForwardOperands<'a, WgpuBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedForward<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = forward_aliases(&operands);
        let plan = plan_convolution_forward::<T, _, R, S>(&operands, parameters, illegal_aliasing)?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::regular_forward(
            operands.input.layout,
            operands.weight.layout,
            operands.output.layout,
            operands.bias.map(|bias| bias.layout),
            parameters,
        )?;
        prepare_forward::<T, R, S>(
            device,
            operands,
            metadata,
            plan.output_elements,
            ConvolutionDirection::Regular,
        )
    }

    fn dispatch_convolution_forward<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_convolution_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: ConvolutionBackwardOperands<'a, WgpuBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedBackward<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = backward_aliases(&operands);
        let plan =
            plan_convolution_backward::<T, _, R, S>(&operands, parameters, illegal_aliasing)?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::regular_backward(
            operands.input.layout,
            operands.weight.layout,
            operands.grad_output.layout,
            parameters,
        )?;
        prepare_backward::<T, R, S>(device, operands, metadata, ConvolutionDirection::Regular)
    }

    fn dispatch_convolution_backward<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_convolution_transposed_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: ConvolutionForwardOperands<'a, WgpuBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedForward<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = forward_aliases(&operands);
        let plan = plan_transposed_convolution_forward::<T, _, R, S>(
            &operands,
            parameters,
            illegal_aliasing,
        )?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::transposed_forward(
            operands.input.layout,
            operands.weight.layout,
            operands.output.layout,
            operands.bias.map(|bias| bias.layout),
            parameters,
        )?;
        prepare_forward::<T, R, S>(
            device,
            operands,
            metadata,
            plan.output_elements,
            ConvolutionDirection::Transposed,
        )
    }

    fn dispatch_convolution_transposed_forward<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedTransposedForward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }

    fn prepare_convolution_transposed_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a WgpuDevice,
        operands: ConvolutionBackwardOperands<'a, WgpuBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedBackward<'a, R, S>> {
        T::validate_capability(device)?;
        validate_spatial_rank::<S>()?;
        let illegal_aliasing = backward_aliases(&operands);
        let plan = plan_transposed_convolution_backward::<T, _, R, S>(
            &operands,
            parameters,
            illegal_aliasing,
        )?;
        plan.validate_address_limit(signed_address_limit())?;
        let metadata = ConvolutionMeta::transposed_backward(
            operands.input.layout,
            operands.weight.layout,
            operands.grad_output.layout,
            parameters,
        )?;
        prepare_backward::<T, R, S>(device, operands, metadata, ConvolutionDirection::Transposed)
    }

    fn dispatch_convolution_transposed_backward<const R: usize, const S: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::PreparedTransposedBackward<'_, R, S>,
    ) -> Result<()> {
        prepared.dispatch(device)
    }
}

fn signed_address_limit() -> usize {
    usize::try_from(i32::MAX).expect("invariant: supported WGPU hosts represent i32 in usize")
}

fn prepare_forward<T, const R: usize, const S: usize>(
    device: &WgpuDevice,
    operands: ConvolutionForwardOperands<'_, WgpuBuffer<T>, R>,
    metadata: ConvolutionMeta,
    output_elements: usize,
    direction: ConvolutionDirection,
) -> Result<PreparedConvolutionKernel>
where
    T: WgpuConvolutionScalar,
{
    let label = match direction {
        ConvolutionDirection::Regular => "hephaestus-convolution-forward",
        ConvolutionDirection::Transposed => "hephaestus-convolution-transposed-forward",
    };
    if output_elements == 0 {
        return Ok(PreparedConvolutionKernel::empty(device, label));
    }
    let bias_mode = if operands.bias.is_some() {
        BiasMode::Present
    } else {
        BiasMode::Absent
    };
    let key = forward_pipeline_key::<S>(direction, bias_mode);
    let pipeline = try_cached_pipeline(
        device,
        (key, TypeId::of::<T>(), WORKGROUP_WIDTH.get()),
        label,
        || forward_shader(T::TYPE_TOKEN, direction, bias_mode, WORKGROUP_WIDTH.get()),
    )?;
    let metadata_buffer = metadata_buffer(device, &metadata)?;
    let bind_group = match operands.bias {
        None => checked_bind_group(
            device,
            &pipeline,
            label,
            &[
                binding(0, operands.input.buffer),
                binding(1, operands.weight.buffer),
                binding(2, operands.output.buffer),
                raw_binding(3, &metadata_buffer),
            ],
        )?,
        Some(bias) => checked_bind_group(
            device,
            &pipeline,
            label,
            &[
                binding(0, operands.input.buffer),
                binding(1, operands.weight.buffer),
                binding(2, bias.buffer),
                binding(3, operands.output.buffer),
                raw_binding(4, &metadata_buffer),
            ],
        )?,
    };
    Ok(PreparedConvolutionKernel::ready(
        device,
        pipeline,
        bind_group,
        metadata_buffer,
        workgroups(output_elements, WORKGROUP_WIDTH)?,
        label,
    ))
}

fn prepare_backward<T, const R: usize, const S: usize>(
    device: &WgpuDevice,
    operands: ConvolutionBackwardOperands<'_, WgpuBuffer<T>, R>,
    base_metadata: ConvolutionMeta,
    direction: ConvolutionDirection,
) -> Result<PreparedConvolutionBackward>
where
    T: WgpuConvolutionScalar,
{
    let input = operands
        .gradients
        .input
        .map(|target| {
            prepare_gradient::<T, R, S>(
                device,
                &operands,
                target,
                base_metadata,
                direction,
                GradientTarget::Input,
            )
        })
        .transpose()?;
    let weight = operands
        .gradients
        .weight
        .map(|target| {
            prepare_gradient::<T, R, S>(
                device,
                &operands,
                target,
                base_metadata,
                direction,
                GradientTarget::Weight,
            )
        })
        .transpose()?;
    let bias = operands
        .gradients
        .bias
        .map(|target| {
            prepare_bias_gradient::<T, R, S>(device, &operands, target, base_metadata, direction)
        })
        .transpose()?;
    Ok(PreparedConvolutionBackward {
        input,
        weight,
        bias,
    })
}

fn prepare_gradient<T, const R: usize, const S: usize>(
    device: &WgpuDevice,
    operands: &ConvolutionBackwardOperands<'_, WgpuBuffer<T>, R>,
    target: hephaestus_core::StridedView<'_, WgpuBuffer<T>, R>,
    base_metadata: ConvolutionMeta,
    direction: ConvolutionDirection,
    gradient: GradientTarget,
) -> Result<PreparedConvolutionKernel>
where
    T: WgpuConvolutionScalar,
{
    let label = gradient_label(direction, gradient);
    let elements = target.layout.checked_size().map_err(layout_error)?;
    if elements == 0 {
        return Ok(PreparedConvolutionKernel::empty(device, label));
    }
    let pipeline = try_cached_pipeline(
        device,
        (
            gradient_pipeline_key::<S>(direction, gradient),
            TypeId::of::<T>(),
            WORKGROUP_WIDTH.get(),
        ),
        label,
        || backward_shader(T::TYPE_TOKEN, direction, gradient, WORKGROUP_WIDTH.get()),
    )?;
    let metadata = base_metadata.with_target(target.layout)?;
    let metadata_buffer = metadata_buffer(device, &metadata)?;
    let other = match gradient {
        GradientTarget::Input => operands.weight.buffer,
        GradientTarget::Weight => operands.input.buffer,
        GradientTarget::Bias => {
            return Err(HephaestusError::InvalidConfiguration {
                message: "bias gradient must use the bias preparation path".to_string(),
            });
        }
    };
    let bind_group = checked_bind_group(
        device,
        &pipeline,
        label,
        &[
            binding(0, operands.grad_output.buffer),
            binding(1, other),
            binding(2, target.buffer),
            raw_binding(3, &metadata_buffer),
        ],
    )?;
    Ok(PreparedConvolutionKernel::ready(
        device,
        pipeline,
        bind_group,
        metadata_buffer,
        workgroups(elements, WORKGROUP_WIDTH)?,
        label,
    ))
}

fn prepare_bias_gradient<T, const R: usize, const S: usize>(
    device: &WgpuDevice,
    operands: &ConvolutionBackwardOperands<'_, WgpuBuffer<T>, R>,
    target: hephaestus_core::StridedView<'_, WgpuBuffer<T>, 1>,
    base_metadata: ConvolutionMeta,
    direction: ConvolutionDirection,
) -> Result<PreparedConvolutionKernel>
where
    T: WgpuConvolutionScalar,
{
    let gradient = GradientTarget::Bias;
    let label = gradient_label(direction, gradient);
    let elements = target.layout.checked_size().map_err(layout_error)?;
    if elements == 0 {
        return Ok(PreparedConvolutionKernel::empty(device, label));
    }
    let pipeline = try_cached_pipeline(
        device,
        (
            gradient_pipeline_key::<S>(direction, gradient),
            TypeId::of::<T>(),
            WORKGROUP_WIDTH.get(),
        ),
        label,
        || backward_shader(T::TYPE_TOKEN, direction, gradient, WORKGROUP_WIDTH.get()),
    )?;
    let metadata = base_metadata.with_target(target.layout)?;
    let metadata_buffer = metadata_buffer(device, &metadata)?;
    let bind_group = checked_bind_group(
        device,
        &pipeline,
        label,
        &[
            binding(0, operands.grad_output.buffer),
            binding(1, target.buffer),
            raw_binding(2, &metadata_buffer),
        ],
    )?;
    Ok(PreparedConvolutionKernel::ready(
        device,
        pipeline,
        bind_group,
        metadata_buffer,
        workgroups(elements, WORKGROUP_WIDTH)?,
        label,
    ))
}
