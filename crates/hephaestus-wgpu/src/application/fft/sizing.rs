//! Geometry and exact buffer/dispatch requirements shared by preflight and preparation.

use hephaestus_core::{DeviceLimits, FftLimit, FftPlan, FftResource, HephaestusError, Result};

use crate::infrastructure::device::WgpuDevice;

use super::{
    kernel::{
        FUSED_MAX_LENGTH, FUSED_WORKGROUP_SIZE, FUSED_WORKGROUP_STORAGE_BYTES, WORKGROUP_SIZE,
    },
    strategy::{Axis, AxisStrategy, fused_grid},
    twiddle::twiddle_elements,
};

/// Exact storage and dispatch requirements of a WGPU dense split-complex FFT.
/// All byte counts describe one buffer. Complex pairs use two buffers of the
/// same size. Per-axis values follow the input shape axis order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct WgpuFftRequirements {
    /// Bytes in each real or imaginary operand buffer.
    pub operand_bytes: u64,
    /// Bytes in each workspace component, reusable by every staged axis.
    pub workspace_bytes: Option<u64>,
    /// Bytes in the staged-radix roots and normalization table.
    pub radix_twiddle_bytes: Option<u64>,
    /// Bytes in the fused-radix roots and normalization table.
    pub fused_twiddle_bytes: Option<u64>,
    /// Bytes per component of a transformed Bluestein kernel on each axis.
    pub chirp_kernel_bytes: [Option<u64>; 3],
    /// Bytes per component of the direct Bluestein factors on each axis.
    pub chirp_factor_bytes: [Option<u64>; 3],
    /// Maximum [X, Y] workgroup grid dispatched for each transformed axis.
    /// Inactive and unit-length axes require [0, 0].
    pub active_workgroups: [[u32; 2]; 3],
}

pub(super) struct Sizing {
    pub(super) dimensions: [usize; 3],
    pub(super) batch: [u32; 3],
    pub(super) strategy: [AxisStrategy; 3],
    pub(super) radix_root_length: usize,
    pub(super) workspace_elements: Option<usize>,
    pub(super) requirements: WgpuFftRequirements,
}

impl Sizing {
    pub(super) fn for_device<T, const R: usize>(
        device: &WgpuDevice,
        plan: &FftPlan<R>,
    ) -> Result<Self> {
        let limits = device.limits();
        Self::new::<T, R>(
            plan,
            device.device_limits(),
            u64::from(limits.max_storage_buffer_binding_size),
            limits.max_compute_workgroups_per_dimension,
        )
    }

    pub(super) fn new<T, const R: usize>(
        plan: &FftPlan<R>,
        limits: DeviceLimits,
        max_storage_binding: u64,
        max_workgroups: u32,
    ) -> Result<Self> {
        plan.validate_address_limit(u32::MAX as usize)?;
        for (limit, value) in [
            (
                FftLimit::WorkgroupSizeX,
                limits.max_compute_workgroup_size_x,
            ),
            (
                FftLimit::InvocationsPerWorkgroup,
                limits.max_compute_invocations_per_workgroup,
            ),
        ] {
            require(
                FftResource::Workgroup,
                u64::from(WORKGROUP_SIZE),
                limit,
                u64::from(value),
            )?;
        }

        // The device-neutral planner proves 1 <= R <= 3 and positive extents.
        let mut dimensions = [1; 3];
        dimensions[..R].copy_from_slice(&plan.shape);
        let axes = [Axis::X, Axis::Y, Axis::Z];
        let mut batch = [0; 3];
        let mut strategy = [AxisStrategy::Identity; 3];
        let mut axis_workspace = [None; 3];
        for (index, axis) in axes.into_iter().enumerate() {
            batch[index] = dimension(
                axis.batch_count(dimensions)
                    .ok_or_else(|| invalid("FFT axis batch count overflows"))?,
                "axis batch count",
            )?;
            strategy[index] = select_axis_strategy(
                plan.active_axes.get(index).copied().unwrap_or(false),
                dimensions[index],
                batch[index],
                limits,
                max_workgroups,
            )?;
            axis_workspace[index] = axis_workspace_elements(dimensions, axis, strategy[index])?;
        }

        let workspace_elements = axis_workspace.into_iter().flatten().max();
        let radix_root_length = radix_root_length(dimensions, strategy);
        let buffer = |resource, elements| -> Result<u64> {
            require(
                resource,
                elements as u64,
                FftLimit::ShaderElements,
                u64::from(u32::MAX),
            )?;
            let bytes = (elements as u64)
                .checked_mul(core::mem::size_of::<T>() as u64)
                .ok_or_else(|| invalid(format!("FFT {resource} byte count overflows")))?;
            require(
                resource,
                bytes,
                FftLimit::BufferBytes,
                limits.max_buffer_size,
            )?;
            require(
                resource,
                bytes,
                FftLimit::StorageBindingBytes,
                max_storage_binding,
            )?;
            Ok(bytes)
        };
        let mut requirements = WgpuFftRequirements {
            operand_bytes: buffer(FftResource::Volume, plan.elements)?,
            workspace_bytes: workspace_elements
                .map(|n| buffer(FftResource::Workspace, n))
                .transpose()?,
            radix_twiddle_bytes: (radix_root_length > 0)
                .then(|| {
                    buffer(
                        FftResource::RadixTwiddle,
                        twiddle_elements(radix_root_length)?,
                    )
                })
                .transpose()?,
            fused_twiddle_bytes: strategy
                .contains(&AxisStrategy::FusedRadix2)
                .then(|| {
                    buffer(
                        FftResource::FusedTwiddle,
                        twiddle_elements(FUSED_MAX_LENGTH)?,
                    )
                })
                .transpose()?,
            chirp_kernel_bytes: [None; 3],
            chirp_factor_bytes: [None; 3],
            active_workgroups: [[0; 2]; 3],
        };
        for (index, axis_strategy) in strategy.into_iter().enumerate() {
            if let AxisStrategy::ChirpZ { n, m } = axis_strategy {
                requirements.chirp_kernel_bytes[index] =
                    Some(buffer(FftResource::ChirpKernel(index), m)?);
                let count = n
                    .checked_add(1)
                    .ok_or_else(|| invalid("FFT direct factor count overflows"))?;
                requirements.chirp_factor_bytes[index] =
                    Some(buffer(FftResource::ChirpFactor(index), count)?);
            }
            if axis_strategy == AxisStrategy::FusedRadix2 {
                requirements.active_workgroups[index] = fused_grid(batch[index], max_workgroups);
            } else if let Some(elements) = axis_workspace[index] {
                let groups = elements.div_ceil(WORKGROUP_SIZE as usize);
                require(
                    FftResource::Dispatch,
                    groups as u64,
                    FftLimit::WorkgroupsPerDimension,
                    u64::from(max_workgroups),
                )?;
                requirements.active_workgroups[index] =
                    [dimension(groups, "dispatch workgroups")?, 1];
            }
        }

        Ok(Self {
            dimensions,
            batch,
            strategy,
            radix_root_length,
            workspace_elements,
            requirements,
        })
    }
}

fn require(
    resource: FftResource,
    required: u64,
    limiting_quantity: FftLimit,
    limit: u64,
) -> Result<()> {
    if required > limit {
        return Err(HephaestusError::FftCapacityExceeded {
            resource,
            required,
            limit,
            limiting_quantity,
        });
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

fn next_power_of_two(n: usize) -> Result<usize> {
    let mut power = 1usize;
    while power < n {
        power = power.checked_mul(2).ok_or_else(|| {
            invalid(format!(
                "FFT length {n} cannot be rounded to a power of two"
            ))
        })?;
    }
    Ok(power)
}

fn axis_strategy_for(n: usize) -> Result<AxisStrategy> {
    if n.is_power_of_two() {
        return Ok(AxisStrategy::StagedRadix2);
    }
    let convolution_len = n
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            invalid(format!(
                "Bluestein convolution length overflows for axis {n}"
            ))
        })?;
    Ok(AxisStrategy::ChirpZ {
        n,
        m: next_power_of_two(convolution_len)?,
    })
}

fn select_fused_strategy(
    strategy: AxisStrategy,
    axis_len: usize,
    batch_count: u32,
    limits: DeviceLimits,
    max_workgroups_per_dimension: u32,
) -> AxisStrategy {
    let grid_capacity = u64::from(max_workgroups_per_dimension)
        .saturating_mul(u64::from(max_workgroups_per_dimension));
    if matches!(strategy, AxisStrategy::StagedRadix2)
        && axis_len > 1
        && axis_len <= FUSED_MAX_LENGTH
        && limits.max_compute_workgroup_size_x >= FUSED_WORKGROUP_SIZE
        && limits.max_compute_invocations_per_workgroup >= FUSED_WORKGROUP_SIZE
        && limits.max_compute_workgroup_storage_size >= FUSED_WORKGROUP_STORAGE_BYTES
        && u64::from(batch_count) <= grid_capacity
    {
        AxisStrategy::FusedRadix2
    } else {
        strategy
    }
}

fn select_axis_strategy(
    active: bool,
    axis_len: usize,
    batch_count: u32,
    limits: DeviceLimits,
    max_workgroups_per_dimension: u32,
) -> Result<AxisStrategy> {
    if !active || axis_len == 1 {
        return Ok(AxisStrategy::Identity);
    }
    Ok(select_fused_strategy(
        axis_strategy_for(axis_len)?,
        axis_len,
        batch_count,
        limits,
        max_workgroups_per_dimension,
    ))
}

pub(super) fn dimension(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid(format!("FFT {name}={value} exceeds the shader u32 domain")))
}

fn axis_workspace_elements(
    dimensions: [usize; 3],
    axis: Axis,
    strategy: AxisStrategy,
) -> Result<Option<usize>> {
    if axis.len(dimensions) == 1 {
        return Ok(None);
    }
    let transform_len = match strategy {
        AxisStrategy::Identity | AxisStrategy::FusedRadix2 => return Ok(None),
        AxisStrategy::StagedRadix2 => axis.len(dimensions),
        AxisStrategy::ChirpZ { m, .. } => m,
    };
    transform_len
        .checked_mul(
            axis.batch_count(dimensions)
                .ok_or_else(|| invalid("FFT axis batch count overflows"))?,
        )
        .ok_or_else(|| {
            invalid(format!(
                "FFT workspace element count overflows for {axis:?}"
            ))
        })
        .map(Some)
}

fn radix_root_length(dimensions: [usize; 3], strategies: [AxisStrategy; 3]) -> usize {
    strategies
        .into_iter()
        .enumerate()
        .filter_map(|(index, strategy)| match strategy {
            AxisStrategy::Identity | AxisStrategy::FusedRadix2 => None,
            AxisStrategy::StagedRadix2 => Some(dimensions[index]),
            AxisStrategy::ChirpZ { m, .. } => Some(m),
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "sizing/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "sizing/device_tests.rs"]
mod device_tests;
