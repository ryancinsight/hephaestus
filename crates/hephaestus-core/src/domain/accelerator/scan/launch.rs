use core::any::TypeId;
use core::marker::PhantomData;
use core::mem::size_of;

use eunomia::Pod;

use super::super::device_api::{DeviceApi, LaunchGeometry};
use super::dialect::{AXIS_SCAN_ENTRY, AxisScanDialect};
use crate::domain::error::{HephaestusError, Result};
use crate::domain::launch::BlockWidth;
use crate::domain::ops::{CombineExpr, IdentityToken};
use crate::domain::scan::{AxisScanMeta, ScanDirection, plan_axis_scan};
use crate::domain::view::StridedView;

/// Marker distinguishing one combining operator's scan kernels in the cache.
struct AxisScanKernel<Op>(PhantomData<Op>);

/// Pipeline-cache identity of one compiled scan kernel.
///
/// A backend admits generic scans by converting this into its own cache key
/// (`impl From<AxisScanKey> for BackendPipelineKey`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisScanKey {
    /// Identity of the combining operator.
    pub marker: TypeId,
    /// Identity of the element scalar.
    pub scalar: TypeId,
    /// Scan direction.
    pub direction: ScanDirection,
    /// Scanned axis.
    pub axis: usize,
    /// Block width the source was specialized to.
    pub width: u32,
}

/// Kernel, geometry, and metadata for one planned scan launch.
pub struct AxisScanLaunch<D: DeviceApi> {
    kernel: D::Kernel,
    meta: AxisScanMeta,
    geometry: LaunchGeometry,
}

/// Resolve the kernel and launch geometry for a rank-2 scan, or `None` when
/// the scan is empty.
///
/// # Errors
///
/// Returns an out-of-range axis, a shape mismatch, an aliased output, a
/// layout validation failure, or the kernel compilation failure.
pub fn plan_axis_scan_launch<D, Op, T>(
    device: &D,
    input: StridedView<'_, D::Buffer<T>, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedView<'_, D::Buffer<T>, 2>,
    width: BlockWidth,
) -> Result<Option<AxisScanLaunch<D>>>
where
    D: DeviceApi,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    Op: CombineExpr<D::Dialect>,
    T: Pod + IdentityToken<Op, D::Dialect>,
{
    let Some(dispatch) = plan_axis_scan(
        input.layout,
        crate::DeviceBuffer::len(input.buffer),
        output.layout,
        crate::DeviceBuffer::len(output.buffer),
        axis,
        direction,
        width,
        D::buffers_alias(input.buffer, output.buffer),
    )?
    else {
        return Ok(None);
    };

    let key = AxisScanKey {
        marker: TypeId::of::<AxisScanKernel<Op>>(),
        scalar: TypeId::of::<T>(),
        direction,
        axis,
        width: width.get(),
    };

    let kernel = device.compile_cached(D::CacheKey::from(key), AXIS_SCAN_ENTRY, || {
        <D::Dialect as AxisScanDialect>::axis_scan_source::<Op, T>(width)
    })?;

    let scalar_bytes =
        u32::try_from(size_of::<T>()).map_err(|_| HephaestusError::DispatchFailed {
            message: format!(
                "scan scalar size exceeds {} shared-memory address range",
                device.backend_name()
            ),
        })?;
    let shared_bytes =
        width
            .get()
            .checked_mul(scalar_bytes)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: "scan shared-memory byte count overflows u32".to_string(),
            })?;

    Ok(Some(AxisScanLaunch {
        kernel,
        meta: dispatch.meta,
        geometry: LaunchGeometry::linear_shared(dispatch.groups, width, shared_bytes),
    }))
}

/// Launch a planned scan against device addresses read at call time, so a
/// caller holding buffer borrows observes writes made after planning.
///
/// # Errors
///
/// Returns the backend launch failure.
pub fn launch_planned_axis_scan<D>(
    device: &D,
    plan: &AxisScanLaunch<D>,
    input: D::DevicePtr,
    output: D::DevicePtr,
) -> Result<()>
where
    D: DeviceApi,
{
    // Argument list mirrors `scan_kernel(AxisScanMeta, const T*, T*)`.
    device.launch(&plan.kernel, plan.geometry, &plan.meta, &[input, output])
}
