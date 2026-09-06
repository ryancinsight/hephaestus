//! Validation and dispatch for the CUDA runtime-rank fusion seam.

use core::ffi::c_void;
use std::any::TypeId;
use std::sync::Arc;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DynamicStridedView, FusedElementwiseOps,
    FusedExpression, FusedReduction, FusedReductionOps, HephaestusError, Result,
};

use crate::CudaDevice;
use crate::application::fusion::source::{
    CudaFusionScalar, FusionLayoutInfo, MAX_FUSION_RANK, elementwise_source, reduction_source,
    validate_expression_source,
};
use crate::application::pipeline::{
    FusionPipelineKey, LaunchConfig, cached_fusion_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;

struct FusedElementwiseKernel;
struct FusedReductionKernel;

struct FusionKernelSpec {
    family: TypeId,
    reduction: Option<FusedReduction>,
    expression: Arc<str>,
    entry: &'static str,
}

fn validate_buffer_owner<T>(
    buffer: &CudaBuffer<T>,
    device: &CudaDevice,
    label: &str,
) -> Result<()> {
    if !buffer.belongs_to(device) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("CUDA fusion {label} buffer belongs to another context"),
        });
    }
    Ok(())
}

fn validate_common<T>(
    device: &CudaDevice,
    inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
    output: DynamicStridedView<'_, CudaBuffer<T>>,
) -> Result<usize> {
    if inputs.is_empty() {
        return Err(HephaestusError::InvalidConfiguration {
            message: "CUDA fusion expression must contain at least one tensor input".to_string(),
        });
    }
    validate_buffer_owner(output.buffer, device, "output")?;
    if output.layout.shape.len() != output.layout.strides.len() {
        return Err(HephaestusError::DispatchFailed {
            message: "CUDA fusion output layout shape and stride ranks differ".to_string(),
        });
    }
    for input in inputs {
        validate_buffer_owner(input.buffer, device, "input")?;
        if input.layout.shape.len() != input.layout.strides.len() {
            return Err(HephaestusError::DispatchFailed {
                message: "CUDA fusion input layout shape and stride ranks differ".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "CUDA fusion output buffer must not alias an input buffer".to_string(),
            });
        }
        input
            .layout
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("CUDA fusion input layout rejected: {error}"),
            })?;
    }
    output
        .layout
        .validate_storage_len(output.buffer.len())
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("CUDA fusion output layout rejected: {error}"),
        })?;
    if !output
        .layout
        .is_injective()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("CUDA fusion output injectivity proof failed: {error}"),
        })?
    {
        return Err(HephaestusError::DispatchFailed {
            message: "CUDA fusion output layout must be injective".to_string(),
        });
    }
    if output.layout.ndim() > MAX_FUSION_RANK {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "CUDA fusion supports rank <= {MAX_FUSION_RANK}, got {}",
                output.layout.ndim()
            ),
        });
    }
    output
        .layout
        .checked_size()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("CUDA fusion output layout rejected: {error}"),
        })
}

fn expression_shape<T: Pod>(
    inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
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
                        "CUDA fusion inputs have incompatible broadcast shapes {:?} and {:?}",
                        shape, input.layout.shape
                    ),
                });
            }
        }
    }
    Ok(shape.into_boxed_slice())
}

fn metadata_for_elementwise<T>(
    inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
    output: DynamicStridedView<'_, CudaBuffer<T>>,
) -> Result<Vec<FusionLayoutInfo>> {
    let mut metadata = Vec::with_capacity(inputs.len() + 1);
    for input in inputs {
        let broadcasted = input
            .layout
            .broadcast(&output.layout.shape)
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("CUDA fusion input is not broadcastable: {error}"),
            })?;
        broadcasted
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("CUDA broadcast fusion input layout rejected: {error}"),
            })?;
        metadata.push(FusionLayoutInfo::from_layout(&broadcasted, None)?);
    }
    metadata.push(FusionLayoutInfo::from_layout(output.layout, None)?);
    Ok(metadata)
}

fn reduction_metadata<T: Pod>(
    inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
    output: DynamicStridedView<'_, CudaBuffer<T>>,
    axis: usize,
) -> Result<(Vec<FusionLayoutInfo>, usize)> {
    let shape = expression_shape(inputs)?;
    if axis >= shape.len() {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "CUDA fusion reduction axis {axis} is out of range for rank {}",
                shape.len()
            ),
        });
    }
    if output.layout.ndim() != shape.len() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "CUDA fusion reduction output rank {} does not match expression rank {}",
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
                    "CUDA fusion reduction output shape {:?} does not match expected {:?}",
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
                    message: format!("CUDA fusion input is not broadcastable: {error}"),
                })?;
        broadcasted
            .validate_storage_len(input.buffer.len())
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("CUDA broadcast fusion input layout rejected: {error}"),
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
    device: &CudaDevice,
    input_buffers: &[&CudaBuffer<T>],
    output: DynamicStridedView<'_, CudaBuffer<T>>,
    metadata: &[FusionLayoutInfo],
    spec: FusionKernelSpec,
    source: impl FnOnce() -> Result<String>,
) -> Result<()>
where
    T: CudaFusionScalar,
{
    let output_len =
        output
            .layout
            .checked_size()
            .map_err(|error| HephaestusError::DispatchFailed {
                message: format!("CUDA fusion output layout rejected: {error}"),
            })?;
    if output_len == 0 {
        return Ok(());
    }
    let rank =
        u32::try_from(output.layout.ndim()).map_err(|_| HephaestusError::InvalidConfiguration {
            message: "CUDA fusion rank exceeds u32 range".to_string(),
        })?;
    let input_count =
        u32::try_from(input_buffers.len()).map_err(|_| HephaestusError::InvalidConfiguration {
            message: "CUDA fusion input count exceeds u32 range".to_string(),
        })?;
    let key = FusionPipelineKey {
        family: spec.family,
        scalar: TypeId::of::<T>(),
        rank,
        input_count,
        reduction: spec.reduction,
        expression: spec.expression,
    };
    let kernel_source = source()?;
    let kernel = cached_fusion_kernel(device, key, spec.entry, move || kernel_source)?;
    let metadata_buffer = device.upload(metadata)?;
    let grid = grid_size(output_len, BlockWidth::DEFAULT)?;
    let mut input_values: Vec<u64> = input_buffers.iter().map(|buffer| buffer.raw()).collect();
    let mut output_value = output.buffer.raw();
    let mut metadata_value = metadata_buffer.raw();
    let mut args: Vec<*mut c_void> = Vec::with_capacity(input_values.len() + 2);
    for value in &mut input_values {
        args.push(core::ptr::from_mut(value).cast());
    }
    args.push(core::ptr::from_mut(&mut output_value).cast());
    args.push(core::ptr::from_mut(&mut metadata_value).cast());
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid, BlockWidth::DEFAULT),
        &mut args,
    )
}

pub(crate) fn fused_elementwise_into<T, E>(
    device: &CudaDevice,
    expression: &E,
    inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
    output: DynamicStridedView<'_, CudaBuffer<T>>,
) -> Result<()>
where
    T: CudaFusionScalar,
    E: FusedExpression<CudaC>,
{
    let expression_source = expression.source();
    validate_expression_source(&expression_source)?;
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
        move || Ok(elementwise_source::<T>(inputs.len(), &source_expression)),
    )
}

pub(crate) fn fused_reduce_into<T, E>(
    device: &CudaDevice,
    expression: &E,
    inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
    reduction: FusedReduction,
    axis: usize,
    output: DynamicStridedView<'_, CudaBuffer<T>>,
) -> Result<()>
where
    T: CudaFusionScalar,
    E: FusedExpression<CudaC>,
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
            message: format!("CUDA fused {reduction:?} reduction does not define an empty axis"),
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
        move || reduction_source::<T>(inputs.len(), &source_expression, reduction),
    )
}

/// Static provider value implementing both runtime fusion roles.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaFusionOps;

impl<T> FusedElementwiseOps<CudaDevice, T> for CudaFusionOps
where
    T: CudaFusionScalar,
{
    type Dialect = CudaC;

    fn fused_elementwise_into<E>(
        &self,
        device: &CudaDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
        output: DynamicStridedView<'_, CudaBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        fused_elementwise_into(device, expression, inputs, output)
    }
}

impl<T> FusedReductionOps<CudaDevice, T> for CudaFusionOps
where
    T: CudaFusionScalar,
{
    type Dialect = CudaC;

    fn fused_reduce_into<E>(
        &self,
        device: &CudaDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, CudaBuffer<T>>],
        reduction: FusedReduction,
        axis: usize,
        output: DynamicStridedView<'_, CudaBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        fused_reduce_into(device, expression, inputs, reduction, axis, output)
    }
}
