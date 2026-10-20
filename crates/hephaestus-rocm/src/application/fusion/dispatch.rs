//! Validation and dispatch for the ROCm runtime-rank fusion seam.

use core::hash::{Hash, Hasher};
use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DynamicStridedView, FusedElementwiseOps,
    FusedExpression, FusedReduction, FusedReductionOps, FusionLayoutInfo, FusionScalar,
    HephaestusError, HipC, MAX_FUSION_RANK, Result, elementwise_source, fusion_expression_shape,
    reduction_source, validate_c_expression_source,
};

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::DevicePtr;
use crate::{RocmBuffer, RocmDevice};

struct FusedElementwiseKernel;
struct FusedReductionKernel;

struct FusionKernelSpec {
    family: TypeId,
    reduction: Option<FusedReduction>,
    expression: String,
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

/// Hash the fusion parameters into the `Copy` pipeline-cache key.
///
/// The expression, scalar, rank, input count, family, and reduction select
/// the generated source, so their hash identifies the compiled kernel exactly
/// like the CUDA backend's structured key.
fn fusion_key<T: 'static>(
    family: TypeId,
    rank: u32,
    input_count: u32,
    reduction: Option<FusedReduction>,
    expression: &str,
) -> PipelineKey {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    family.hash(&mut hasher);
    TypeId::of::<T>().hash(&mut hasher);
    rank.hash(&mut hasher);
    input_count.hash(&mut hasher);
    reduction
        .map(|reduction| format!("{reduction:?}"))
        .hash(&mut hasher);
    expression.hash(&mut hasher);
    PipelineKey::Fusion(hasher.finish())
}

fn submit<T>(
    device: &RocmDevice,
    input_buffers: &[&RocmBuffer<T>],
    output: DynamicStridedView<'_, RocmBuffer<T>>,
    metadata: &[FusionLayoutInfo],
    spec: FusionKernelSpec,
    source: impl FnOnce() -> Result<String>,
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
    let key = fusion_key::<T>(
        spec.family,
        rank,
        input_count,
        spec.reduction,
        &spec.expression,
    );
    let kernel_source = source()?;
    let kernel = cached_kernel(device, key, spec.entry, move || kernel_source)?;
    let metadata_buffer = device.upload(metadata)?;
    let grid = grid_size(output_len, BlockWidth::DEFAULT)?;
    let mut device_ptrs: Vec<DevicePtr> = input_buffers.iter().map(|buffer| buffer.raw()).collect();
    let mut output_value: DevicePtr = output.buffer.raw();
    let mut metadata_value: DevicePtr = metadata_buffer.raw();
    let mut args: Vec<*mut core::ffi::c_void> = device_ptrs
        .iter_mut()
        .map(|ptr| ptr as *mut DevicePtr as *mut core::ffi::c_void)
        .collect();
    args.push(&mut output_value as *mut DevicePtr as *mut core::ffi::c_void);
    args.push(&mut metadata_value as *mut DevicePtr as *mut core::ffi::c_void);
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
    let input_buffers = inputs.iter().map(|input| input.buffer).collect::<Vec<_>>();
    submit(
        device,
        &input_buffers,
        output,
        &metadata,
        FusionKernelSpec {
            family: TypeId::of::<FusedElementwiseKernel>(),
            reduction: None,
            expression: source_expression.clone(),
            entry: "fused_elementwise_kernel",
        },
        move || Ok(elementwise_source::<HipC, T>(inputs.len(), &source_expression)),
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
            expression: format!("{reduction:?}:{source_expression}"),
            entry: "fused_reduction_kernel",
        },
        move || Ok(reduction_source::<HipC, T>(inputs.len(), &source_expression, reduction)),
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
