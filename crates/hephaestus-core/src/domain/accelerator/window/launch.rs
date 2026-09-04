//! Operation selection, kernel compilation, and the device launch call.

use core::any::TypeId;

use eunomia::Pod;
use leto::Layout;

use crate::domain::accelerator::device_api::{DeviceApi, LaunchGeometry};
use crate::domain::dialect::DialectScalar;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::pooling::PoolingMode;

use super::key::{WindowKey, WindowOperation};
use super::metadata;
use super::prepared::WindowLaunch;
use super::source::{self, WindowDialect};

pub(super) fn pooling_forward_operation(mode: PoolingMode) -> WindowOperation {
    match mode {
        PoolingMode::Maximum => WindowOperation::PoolingForwardMaximum,
        PoolingMode::Average => WindowOperation::PoolingForwardAverage,
    }
}

pub(super) fn pooling_backward_operation(mode: PoolingMode) -> WindowOperation {
    match mode {
        PoolingMode::Maximum => WindowOperation::PoolingBackwardMaximum,
        PoolingMode::Average => WindowOperation::PoolingBackwardAverage,
    }
}

pub(super) fn validate_spatial_rank<const S: usize>() -> Result<()> {
    if !(1..=3).contains(&S) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "C-family window operations support spatial ranks 1 through 3, got {S}"
            ),
        });
    }
    Ok(())
}

pub(super) fn elements<const R: usize>(layout: &Layout<R>, name: &str) -> Result<usize> {
    layout
        .checked_size()
        .map_err(|error| HephaestusError::InvalidConfiguration {
            message: format!("{name} layout rejected: {error}"),
        })
}

pub(super) fn prepare_launch<D, T>(
    device: &D,
    operation: WindowOperation,
    metadata: metadata::WindowMeta,
    elements: usize,
    spatial_rank: usize,
) -> Result<Option<WindowLaunch<D>>>
where
    D: DeviceApi,
    D::CacheKey: From<WindowKey>,
    D::Dialect: WindowDialect,
    T: DialectScalar<D::Dialect> + Pod + 'static,
{
    if elements == 0 {
        return Ok(None);
    }
    let width = crate::domain::launch::BlockWidth::DEFAULT;
    let work_items = u64::try_from(elements).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("window element count {elements} exceeds u64 range"),
    })?;
    let groups = width.checked_covering_blocks(work_items).ok_or_else(|| {
        HephaestusError::DispatchFailed {
            message: format!("window element count {elements} exceeds u32 grid range"),
        }
    })?;
    let key = WindowKey {
        operation,
        scalar: TypeId::of::<T>(),
        spatial_rank,
    };
    let kernel = device.compile_cached(D::CacheKey::from(key), source::WINDOW_ENTRY, || {
        <D::Dialect as WindowDialect>::window_source::<T>(operation, width)
    })?;
    Ok(Some(WindowLaunch {
        kernel,
        metadata,
        geometry: LaunchGeometry::linear(groups, width),
    }))
}

pub(super) fn launch_window<D: DeviceApi>(
    device: &D,
    launch: &WindowLaunch<D>,
    operands: &[D::DevicePtr],
) -> Result<()> {
    device.launch(&launch.kernel, launch.geometry, &launch.metadata, operands)
}
