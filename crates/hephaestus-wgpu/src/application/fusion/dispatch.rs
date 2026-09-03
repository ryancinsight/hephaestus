//! Validation and dispatch for the runtime-rank fusion seam.

use std::any::TypeId;
use std::sync::Arc;

use bytemuck::Pod;
use hephaestus_core::{
    ComputeDevice, DeviceBuffer, DynamicStridedView, FusedElementwiseOps, FusedExpression,
    FusedReduction, FusedReductionOps, HephaestusError, Result, Wgsl,
};

use crate::application::bindings::BindGroupEntries;
use crate::application::fusion::source::{
    FusionLayoutInfo, MAX_FUSION_RANK, WgpuFusionScalar, elementwise_source, reduction_source,
    validate_expression_source,
};
use crate::application::pipeline::{try_cached_fusion_pipeline, workgroups};
use crate::application::prepared::{checked_bind_group, checked_submit, validate_buffer_owner};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{FusionPipelineKey, WgpuDevice};

struct FusedElementwiseKernel;
struct FusedReductionKernel;

struct FusionKernelSpec<'a> {
    expression: &'a str,
    family: TypeId,
    label: &'static str,
}

fn validate_binding_count(device: &WgpuDevice, input_count: usize) -> Result<()> {
    let required =
        input_count
            .checked_add(2)
            .ok_or_else(|| HephaestusError::InvalidConfiguration {
                message: "fusion binding count overflows usize".to_string(),
            })?;
    let limit =
        usize::try_from(device.limits().max_storage_buffers_per_shader_stage).map_err(|_| {
            HephaestusError::InvalidConfiguration {
                message: "WGPU storage-buffer binding limit does not fit usize".to_string(),
            }
        })?;
    if required > limit {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "fusion requires {required} storage bindings, device limit is {limit}"
            ),
        });
    }
    Ok(())
}

fn validate_common<T>(
    device: &WgpuDevice,
    inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
    output: DynamicStridedView<'_, WgpuBuffer<T>>,
) -> Result<usize> {
    if inputs.is_empty() {
        return Err(HephaestusError::InvalidConfiguration {
            message: "fusion expression must contain at least one tensor input".to_string(),
        });
    }
    validate_binding_count(device, inputs.len())?;
    validate_buffer_owner(output.buffer, device, "fusion output")?;
    let validate_layout = |layout: &leto::LayoutDyn, label: &str| {
        if layout.shape.len() != layout.strides.len() {
            return Err(HephaestusError::DispatchFailed {
                message: format!("{label} shape and stride ranks differ"),
            });
        }
        Ok(())
    };
    validate_layout(output.layout, "fusion output layout")?;
    for input in inputs {
        validate_buffer_owner(input.buffer, device, "fusion input")?;
        validate_layout(input.layout, "fusion input layout")?;
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "fusion output buffer must not alias an input buffer".to_string(),
            });
        }
        input
            .layout
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("fusion input layout rejected: {error}"),
            })?;
    }
    output
        .layout
        .validate_storage_len(output.buffer.len())
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("fusion output layout rejected: {error}"),
        })?;
    if !output
        .layout
        .is_injective()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("fusion output injectivity proof failed: {error}"),
        })?
    {
        return Err(HephaestusError::DispatchFailed {
            message: "fusion output layout must be injective".to_string(),
        });
    }
    if output.layout.ndim() > MAX_FUSION_RANK {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "WGPU fusion supports rank <= {MAX_FUSION_RANK}, got {}",
                output.layout.ndim()
            ),
        });
    }
    output
        .layout
        .checked_size()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("fusion output layout rejected: {error}"),
        })
}

fn metadata_for_elementwise<T>(
    inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
    output: DynamicStridedView<'_, WgpuBuffer<T>>,
) -> Result<Vec<FusionLayoutInfo>> {
    let output_len =
        output
            .layout
            .checked_size()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("fusion output layout rejected: {error}"),
            })?;
    let mut metadata = Vec::with_capacity(inputs.len() + 1);
    for input in inputs {
        let broadcasted = input
            .layout
            .broadcast(&output.layout.shape)
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("fusion input is not broadcastable: {error}"),
            })?;
        broadcasted
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("broadcast fusion input layout rejected: {error}"),
            })?;
        metadata.push(FusionLayoutInfo::from_layout(&broadcasted, None)?);
    }
    metadata.push(FusionLayoutInfo::from_layout(output.layout, Some((0, 1)))?);
    debug_assert_eq!(
        metadata.last().map(|info| info.length),
        Some(u32::try_from(output_len).expect("invariant: metadata conversion validated length"))
    );
    Ok(metadata)
}

fn expression_shape<T: Pod>(
    inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
) -> Result<Box<[usize]>> {
    let rank = inputs
        .iter()
        .map(|input| input.layout.ndim())
        .max()
        .unwrap_or(0);
    let mut shape = vec![1usize; rank];
    for input in inputs {
        let shift = rank - input.layout.ndim();
        for (axis, &dimension) in input.layout.shape.iter().enumerate() {
            let target_axis = axis + shift;
            let target = &mut shape[target_axis];
            if *target == 1 {
                *target = dimension;
            } else if dimension != 1 && *target != dimension {
                return Err(HephaestusError::DispatchFailed {
                    message: format!(
                        "fusion inputs have incompatible broadcast shapes {:?} and {:?}",
                        shape, input.layout.shape
                    ),
                });
            }
        }
    }
    Ok(shape.into_boxed_slice())
}

fn reduction_metadata<T>(
    inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
    output: DynamicStridedView<'_, WgpuBuffer<T>>,
    axis: usize,
) -> Result<(Vec<FusionLayoutInfo>, usize)>
where
    T: Pod,
{
    let shape = expression_shape(inputs)?;
    if axis >= shape.len() {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "fusion reduction axis {axis} is out of range for rank {}",
                shape.len()
            ),
        });
    }
    if output.layout.ndim() != shape.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "fusion reduction output rank {} does not match expression rank {}",
                output.layout.ndim(),
                shape.len()
            ),
        });
    }
    for (output_axis, (&actual, &expected)) in
        output.layout.shape.iter().zip(shape.iter()).enumerate()
    {
        let expected = if output_axis == axis { 1 } else { expected };
        if actual != expected {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "fusion reduction output shape {:?} does not match expected {:?}",
                    output.layout.shape,
                    shape
                        .iter()
                        .enumerate()
                        .map(|(index, &extent)| if index == axis { 1 } else { extent })
                        .collect::<Vec<_>>()
                ),
            });
        }
    }
    let axis_length = shape[axis];
    let mut metadata = Vec::with_capacity(inputs.len() + 1);
    for input in inputs {
        let broadcasted =
            input
                .layout
                .broadcast(&shape)
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("fusion input is not broadcastable: {error}"),
                })?;
        broadcasted
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("broadcast fusion input layout rejected: {error}"),
            })?;
        metadata.push(FusionLayoutInfo::from_layout(&broadcasted, None)?);
    }
    metadata.push(FusionLayoutInfo::from_layout(
        output.layout,
        Some((axis, axis_length)),
    )?);
    Ok((metadata, axis_length))
}

fn submit<T>(
    device: &WgpuDevice,
    input_buffers: &[&WgpuBuffer<T>],
    output: DynamicStridedView<'_, WgpuBuffer<T>>,
    metadata: &[FusionLayoutInfo],
    spec: FusionKernelSpec<'_>,
    source: impl FnOnce() -> Result<String>,
) -> Result<()>
where
    T: WgpuFusionScalar,
{
    let output_len =
        output
            .layout
            .checked_size()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("fusion output layout rejected: {error}"),
            })?;
    if output_len == 0 {
        return Ok(());
    }
    let source_key = Arc::<str>::from(spec.expression);
    let key = FusionPipelineKey {
        family: spec.family,
        scalar: TypeId::of::<T>(),
        rank: u32::try_from(output.layout.ndim()).map_err(|_| {
            HephaestusError::InvalidConfiguration {
                message: "fusion rank exceeds u32 range".to_string(),
            }
        })?,
        input_count: u32::try_from(input_buffers.len()).map_err(|_| {
            HephaestusError::InvalidConfiguration {
                message: "fusion input count exceeds u32 range".to_string(),
            }
        })?,
        expression: source_key,
    };
    let shader_source = source()?;
    let pipeline = try_cached_fusion_pipeline(device, key, spec.label, || shader_source)?;
    let metadata_buffer = device.upload(metadata)?;
    let output_binding =
        u32::try_from(input_buffers.len()).map_err(|_| HephaestusError::InvalidConfiguration {
            message: "fusion output binding exceeds u32 range".to_string(),
        })?;
    let layout_binding =
        output_binding
            .checked_add(1)
            .ok_or_else(|| HephaestusError::InvalidConfiguration {
                message: "fusion layout binding exceeds u32 range".to_string(),
            })?;
    let mut entries = BindGroupEntries::with_capacity(input_buffers.len() + 2);
    for (binding, input) in input_buffers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: u32::try_from(binding).map_err(|_| HephaestusError::InvalidConfiguration {
                message: "fusion input binding exceeds u32 range".to_string(),
            })?,
            resource: input.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: output_binding,
        resource: output.buffer.as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: layout_binding,
        resource: metadata_buffer.as_entire_binding(),
    });
    let bind_group = checked_bind_group(device, &pipeline, spec.label, &entries)?;
    let groups = workgroups(output_len, hephaestus_core::BlockWidth::DEFAULT)?;
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(spec.label),
        });
    crate::application::pipeline::encode_compute_pass(
        &mut encoder,
        &pipeline,
        &bind_group,
        groups,
        spec.label,
    );
    checked_submit(device, spec.label, encoder)
}

pub(crate) fn fused_elementwise_into<T, E>(
    device: &WgpuDevice,
    expression: &E,
    inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
    output: DynamicStridedView<'_, WgpuBuffer<T>>,
) -> Result<()>
where
    T: WgpuFusionScalar,
    E: FusedExpression<Wgsl>,
{
    let expression_source = expression.source();
    validate_expression_source(&expression_source)?;
    validate_common(device, inputs, output)?;
    let metadata = metadata_for_elementwise(inputs, output)?;
    let source_expression = expression_source.into_owned();
    let pipeline_expression = source_expression.clone();
    let input_buffers = inputs.iter().map(|input| input.buffer).collect::<Vec<_>>();
    submit(
        device,
        &input_buffers,
        output,
        &metadata,
        FusionKernelSpec {
            expression: &pipeline_expression,
            family: TypeId::of::<FusedElementwiseKernel>(),
            label: "hephaestus-fused-elementwise",
        },
        move || Ok(elementwise_source::<T>(inputs.len(), &source_expression)),
    )
}

pub(crate) fn fused_reduce_into<T, E>(
    device: &WgpuDevice,
    expression: &E,
    inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
    reduction: FusedReduction,
    axis: usize,
    output: DynamicStridedView<'_, WgpuBuffer<T>>,
) -> Result<()>
where
    T: WgpuFusionScalar,
    E: FusedExpression<Wgsl>,
{
    let expression_source = expression.source();
    validate_expression_source(&expression_source)?;
    validate_common(device, inputs, output)?;
    let (metadata, axis_length) = reduction_metadata(inputs, output, axis)?;
    if matches!(
        reduction,
        FusedReduction::Mean | FusedReduction::Maximum | FusedReduction::Minimum
    ) && axis_length == 0
    {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("fused {reduction:?} reduction does not define an empty axis"),
        });
    }
    let source_expression = expression_source.into_owned();
    let pipeline_expression = format!("{reduction:?}:{source_expression}");
    let shader_expression = source_expression.clone();
    let mut empty_input_buffers = Vec::with_capacity(inputs.len());
    for input in inputs {
        empty_input_buffers.push(
            (input.buffer.len() == 0)
                .then(|| device.alloc_zeroed::<T>(1))
                .transpose()?,
        );
    }
    let input_buffers = inputs
        .iter()
        .zip(&empty_input_buffers)
        .map(|(input, replacement)| replacement.as_ref().map_or(input.buffer, |buffer| buffer))
        .collect::<Vec<_>>();
    submit(
        device,
        &input_buffers,
        output,
        &metadata,
        FusionKernelSpec {
            expression: &pipeline_expression,
            family: TypeId::of::<FusedReductionKernel>(),
            label: "hephaestus-fused-reduction",
        },
        move || reduction_source::<T>(inputs.len(), &shader_expression, reduction),
    )
}

/// Static provider value implementing both runtime fusion roles.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuFusionOps;

impl<T> FusedElementwiseOps<WgpuDevice, T> for WgpuFusionOps
where
    T: WgpuFusionScalar,
{
    type Dialect = Wgsl;

    fn fused_elementwise_into<E>(
        &self,
        device: &WgpuDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
        output: DynamicStridedView<'_, WgpuBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        fused_elementwise_into(device, expression, inputs, output)
    }
}

impl<T> FusedReductionOps<WgpuDevice, T> for WgpuFusionOps
where
    T: WgpuFusionScalar,
{
    type Dialect = Wgsl;

    fn fused_reduce_into<E>(
        &self,
        device: &WgpuDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, WgpuBuffer<T>>],
        reduction: FusedReduction,
        axis: usize,
        output: DynamicStridedView<'_, WgpuBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        fused_reduce_into(device, expression, inputs, reduction, axis, output)
    }
}
