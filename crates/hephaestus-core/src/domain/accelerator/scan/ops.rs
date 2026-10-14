use core::marker::PhantomData;

use eunomia::Pod;
use leto::Layout;

use super::super::device_api::DeviceApi;
use super::dialect::AxisScanDialect;
use super::launch::{AxisScanKey, AxisScanLaunch, launch_planned_axis_scan, plan_axis_scan_launch};
use crate::domain::dialect::DialectScalar;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::launch::BlockWidth;
use crate::domain::ops::{CombineExpr, IdentityToken, OpIdentity};
use crate::domain::scan::{ScanDirection, ScanOps};
use crate::domain::view::StridedView;

/// The [`ScanOps`] seam implemented once for every [`DeviceApi`] backend.
///
/// A zero-sized marker parameterized by the device it scans, so a bound of
/// `S: ScanOps<D, T>` still monomorphizes to that backend's own kernel
/// dispatch at no runtime cost.
pub struct AxisScanOps<D>(PhantomData<fn() -> D>);

// Hand-written rather than derived: a derive would bound each impl on `D`,
// but the marker holds no `D` value and every operation is available for any
// device.
impl<D> Clone for AxisScanOps<D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for AxisScanOps<D> {}

impl<D> Default for AxisScanOps<D> {
    #[inline]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D> core::fmt::Debug for AxisScanOps<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("AxisScanOps")
    }
}

impl<D> AxisScanOps<D> {
    /// The scan seam for device `D`.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// A scan bound to one input/output pair.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound operands between dispatches are observed (the seam's
/// rebind contract). An empty scan prepares to a no-op.
pub struct PreparedAxisScan<'op, D: DeviceApi, T: Pod> {
    input: &'op D::Buffer<T>,
    output: &'op D::Buffer<T>,
    plan: Option<AxisScanLaunch<D>>,
}

// `D: 'static` is required because a prepared scan borrows the device's own
// buffers for `'op`; a GPU device handle owns refcounted context state and is
// always `'static`, and `T: Pod` already implies it for the scalar.
impl<D, T> ScanOps<D, T> for AxisScanOps<D>
where
    D: DeviceApi + 'static,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    T: Pod + DialectScalar<D::Dialect>,
{
    type Dialect = D::Dialect;
    type PreparedScan<'op, const N: usize>
        = PreparedAxisScan<'op, D, T>
    where
        Self: 'op,
        D: 'op,
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, D::Buffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{} scan supports rank 2 only, got rank {N}",
                    device.backend_name()
                ),
            });
        }
        // The rank guard above proves N == 2, so the rank-2 components are
        // total; rebuilding a Layout<2> avoids reinterpreting &Layout<N> and
        // keeps the featureless build under forbid(unsafe_code).
        let input_layout = Layout::try_new(
            [input.layout.shape()[0], input.layout.shape()[1]],
            [input.layout.strides()[0], input.layout.strides()[1]],
            input.layout.offset(),
        )
        .expect("invariant: rank guard proves the rank-2 components are total");
        let output_layout = Layout::try_new(
            [output.layout.shape()[0], output.layout.shape()[1]],
            [output.layout.strides()[0], output.layout.strides()[1]],
            output.layout.offset(),
        )
        .expect("invariant: rank guard proves the rank-2 components are total");
        let plan = plan_axis_scan_launch::<D, Op, T>(
            device,
            StridedView::new(input.buffer, &input_layout),
            axis,
            direction,
            StridedView::new(output.buffer, &output_layout),
            BlockWidth::DEFAULT,
        )?;
        Ok(PreparedAxisScan {
            input: input.buffer,
            output: output.buffer,
            plan,
        })
    }

    fn dispatch_scan<const N: usize>(
        &self,
        device: &D,
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        let Some(plan) = &prepared.plan else {
            return Ok(());
        };
        launch_planned_axis_scan(
            device,
            plan,
            D::device_ptr(prepared.input),
            D::device_ptr(prepared.output),
        )
    }
}
