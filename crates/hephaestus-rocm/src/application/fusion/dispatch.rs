//! Validation and dispatch for the ROCm runtime-rank fusion seam.
//!
//! Kernel source, layout metadata, and the scalar vocabulary are owned by
//! `hephaestus-core` and monomorphized here over [`HipC`]; the HIP
//! instantiation compiles byte-identical kernels to the CUDA one from that
//! single generator. This module owns validation, pipeline caching, and
//! launch only.

use std::any::TypeId;
use std::sync::Arc;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DynamicStridedView, FusedElementwiseOps,
    FusedExpression, FusedReduction, FusedReductionOps, FusionLayoutInfo, FusionPipelineKey,
    FusionScalar, HephaestusError, HipC, MAX_FUSION_RANK, Result, elementwise_source,
    fusion_expression_shape, reduction_source, validate_c_expression_source,
};

use crate::application::pipeline::{LaunchConfig, cached_fusion_kernel, grid_size, launch_kernel};
use crate::infrastructure::DevicePtr;
use crate::{RocmBuffer, RocmDevice};

struct FusedElementwiseKernel;
struct FusedReductionKernel;

struct FusionKernelSpec {
    family: TypeId,
    reduction: Option<FusedReduction>,
    expression: Arc<str>,
    entry: &'static str,
}

fn validate_buffer_owner<T>(
    buffer: &RocmBuffer<T>,
    device: &RocmDevice,
    label: &str,
) -> Result<()> {
    if !buffer.belongs_to(device) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("ROCm fusion {label} buffer belongs to another context"),
        });
    }
    Ok(())
}

fn validate_common<T: Pod>(
    device: &RocmDevice,
    inputs: &[DynamicStridedView<'_, RocmBuffer<T>>],
    output: DynamicStridedView<'_, RocmBuffer<T>>,
) -> Result<usize> {
    if inputs.is_empty() {
        return Err(HephaestusError::InvalidConfiguration {
            message: "ROCm fusion expression must contain at least one tensor input".to_string(),
        });
    }
    validate_buffer_owner(output.buffer, device, "output")?;
    if output.layout.shape.len() != output.layout.strides.len() {
        return Err(HephaestusError::DispatchFailed {
            message: "ROCm fusion output layout shape and stride ranks differ".to_string(),
        });
    }
    for input in inputs {
        validate_buffer_owner(input.buffer, device, "input")?;
        if input.layout.shape.len() != input.layout.strides.len() {
            return Err(HephaestusError::DispatchFailed {
                message: "ROCm fusion input layout shape and stride ranks differ".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "ROCm fusion output buffer must not alias an input buffer".to_string(),
            });
        }
        input
            .layout
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("ROCm fusion input layout rejected: {error}"),
            })?;
    }
    output
        .layout
        .validate_storage_len(output.buffer.len())
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("ROCm fusion output layout rejected: {error}"),
        })?;
    if !output
        .layout
        .is_injective()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("ROCm fusion output injectivity proof failed: {error}"),
        })?
    {
        return Err(HephaestusError::DispatchFailed {
            message: "ROCm fusion output layout must be injective".to_string(),
        });
    }
    if output.layout.ndim() > MAX_FUSION_RANK {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "ROCm fusion supports rank <= {MAX_FUSION_RANK}, got {}",
                output.layout.ndim()
            ),
        });
    }
    output
        .layout
        .checked_size()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("ROCm fusion output layout rejected: {error}"),
        })
}

fn metadata_for_elementwise<T: Pod>(
    inputs: &[DynamicStridedView<'_, RocmBuffer<T>>],
    output: DynamicStridedView<'_, RocmBuffer<T>>,
) -> Result<Vec<FusionLayoutInfo>> {
    let mut metadata = Vec::with_capacity(inputs.len() + 1);
    for input in inputs {
        let broadcasted = input
            .layout
            .broadcast(&output.layout.shape)
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("ROCm fusion input is not broadcastable: {error}"),
            })?;
        broadcasted
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("ROCm broadcast fusion input layout rejected: {error}"),
            })?;
        metadata.push(FusionLayoutInfo::from_layout(&broadcasted, None)?);
    }
    metadata.push(FusionLayoutInfo::from_layout(output.layout, None)?);
    Ok(metadata)
}

fn reduction_metadata<T: Pod>(
    inputs: &[DynamicStridedView<'_, RocmBuffer<T>>],
    output: DynamicStridedView<'_, RocmBuffer<T>>,
    axis: usize,
) -> Result<(Vec<FusionLayoutInfo>, usize)> {
    let shape = fusion_expression_shape(inputs)?;
    if axis >= shape.len() {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "ROCm fusion reduction axis {axis} is out of range for rank {}",
                shape.len()
            ),
        });
    }
    if output.layout.ndim() != shape.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "ROCm fusion reduction output rank {} does not match expression rank {}",
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
                    "ROCm fusion reduction output shape {:?} does not match expected {:?}",
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
                    message: format!("ROCm fusion input is not broadcastable: {error}"),
                })?;
        broadcasted
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("ROCm broadcast fusion input layout rejected: {error}"),
            })?;
        metadata.push(FusionLayoutInfo::from_layout(
            &broadcasted,
            Some((axis, axis_length)),
        )?);
    }
    metadata.push(FusionLayoutInfo::from_layout(
        output.layout,
        Some((axis, axis_length)),
    )?);
    Ok((metadata, axis_length))
}

fn submit<T>(
    device: &RocmDevice,
    input_buffers: &[&RocmBuffer<T>],
    output: DynamicStridedView<'_, RocmBuffer<T>>,
    metadata: &[FusionLayoutInfo],
    spec: FusionKernelSpec,
    source: impl FnOnce() -> String,
) -> Result<()>
where
    T: FusionScalar<HipC>,
{
    let output_len =
        output
            .layout
            .checked_size()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("ROCm fusion output layout rejected: {error}"),
            })?;
    if output_len == 0 {
        return Ok(());
    }
    let rank =
        u32::try_from(output.layout.ndim()).map_err(|_| HephaestusError::InvalidConfiguration {
            message: "ROCm fusion rank exceeds u32 range".to_string(),
        })?;
    let input_count =
        u32::try_from(input_buffers.len()).map_err(|_| HephaestusError::InvalidConfiguration {
            message: "ROCm fusion input count exceeds u32 range".to_string(),
        })?;
    let key = FusionPipelineKey {
        family: spec.family,
        scalar: TypeId::of::<T>(),
        rank,
        input_count,
        reduction: spec.reduction,
        expression: spec.expression,
    };
    let kernel = cached_fusion_kernel(device, key, spec.entry, source)?;
    let metadata_buffer = device.upload(metadata)?;
    let grid = grid_size(output_len, BlockWidth::DEFAULT)?;
    let mut input_values: Vec<DevicePtr> =
        input_buffers.iter().map(|buffer| buffer.raw()).collect();
    let mut output_value = output.buffer.raw();
    let mut metadata_value = metadata_buffer.raw();
    let mut args: Vec<*mut core::ffi::c_void> = Vec::with_capacity(input_values.len() + 2);
    for value in &mut input_values {
        args.push((value as *mut DevicePtr).cast());
    }
    args.push((&mut output_value as *mut DevicePtr).cast());
    args.push((&mut metadata_value as *mut DevicePtr).cast());
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, BlockWidth::DEFAULT),
        &mut args,
    )
}

pub(crate) fn fused_elementwise_into<T, E>(
    device: &RocmDevice,
    expression: &E,
    inputs: &[DynamicStridedView<'_, RocmBuffer<T>>],
    output: DynamicStridedView<'_, RocmBuffer<T>>,
) -> Result<()>
where
    T: FusionScalar<HipC>,
    E: FusedExpression<HipC>,
{
    let expression_source = expression.source();
    validate_c_expression_source(&expression_source)?;
    validate_common(device, inputs, output)?;
    let metadata = metadata_for_elementwise(inputs, output)?;
    let source_expression = expression_source.into_owned();
    let expression_key = Arc::<str>::from(source_expression.as_str());
    let input_buffers = inputs.iter().map(|input| input.buffer).collect::<Vec<_>>();
    submit(
        device,
        &input_buffers,
        output,
        &metadata,
        FusionKernelSpec {
            family: TypeId::of::<FusedElementwiseKernel>(),
            reduction: None,
            expression: expression_key,
            entry: "fused_elementwise_kernel",
        },
        move || elementwise_source::<HipC, T>(inputs.len(), &source_expression),
    )
}

pub(crate) fn fused_reduce_into<T, E>(
    device: &RocmDevice,
    expression: &E,
    inputs: &[DynamicStridedView<'_, RocmBuffer<T>>],
    reduction: FusedReduction,
    axis: usize,
    output: DynamicStridedView<'_, RocmBuffer<T>>,
) -> Result<()>
where
    T: FusionScalar<HipC>,
    E: FusedExpression<HipC>,
{
    let expression_source = expression.source();
    validate_c_expression_source(&expression_source)?;
    validate_common(device, inputs, output)?;
    let (metadata, axis_length) = reduction_metadata(inputs, output, axis)?;
    if matches!(
        reduction,
        FusedReduction::Mean | FusedReduction::Maximum | FusedReduction::Minimum
    ) && axis_length == 0
    {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("ROCm fused {reduction:?} reduction does not define an empty axis"),
        });
    }
    let source_expression = expression_source.into_owned();
    let expression_key = Arc::<str>::from(format!("{reduction:?}:{source_expression}"));
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
            family: TypeId::of::<FusedReductionKernel>(),
            reduction: Some(reduction),
            expression: expression_key,
            entry: "fused_reduction_kernel",
        },
        move || reduction_source::<HipC, T>(inputs.len(), &source_expression, reduction),
    )
}

/// Static provider value implementing both runtime fusion roles.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmFusionOps;

impl<T> FusedElementwiseOps<RocmDevice, T> for RocmFusionOps
where
    T: FusionScalar<HipC>,
{
    type Dialect = HipC;

    fn fused_elementwise_into<E>(
        &self,
        device: &RocmDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, RocmBuffer<T>>],
        output: DynamicStridedView<'_, RocmBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        fused_elementwise_into(device, expression, inputs, output)
    }
}

impl<T> FusedReductionOps<RocmDevice, T> for RocmFusionOps
where
    T: FusionScalar<HipC>,
{
    type Dialect = HipC;

    fn fused_reduce_into<E>(
        &self,
        device: &RocmDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, RocmBuffer<T>>],
        reduction: FusedReduction,
        axis: usize,
        output: DynamicStridedView<'_, RocmBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        fused_reduce_into(device, expression, inputs, reduction, axis, output)
    }
}
