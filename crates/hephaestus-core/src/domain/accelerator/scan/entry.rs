use eunomia::Pod;
use leto::Layout;

use super::super::device_api::DeviceApi;
use super::dialect::AxisScanDialect;
use super::launch::{AxisScanKey, launch_planned_axis_scan, plan_axis_scan_launch};
use crate::domain::error::Result;
use crate::domain::launch::BlockWidth;
use crate::domain::ops::{CombineExpr, CumProdOp, CumSumOp, IdentityToken};
use crate::domain::scan::ScanDirection;
use crate::domain::strided::map_layout_err;
use crate::domain::view::StridedView;

/// Scan a rank-2 strided operand along `axis`, preserving the input shape.
///
/// # Errors
///
/// Returns an out-of-range axis, a shape mismatch, an aliased output, a
/// layout validation failure, or the backend dispatch failure.
pub fn scan_axis_into<D, Op, T>(
    device: &D,
    input: StridedView<'_, D::Buffer<T>, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedView<'_, D::Buffer<T>, 2>,
    width: BlockWidth,
) -> Result<()>
where
    D: DeviceApi,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    Op: CombineExpr<D::Dialect>,
    T: Pod + IdentityToken<Op, D::Dialect>,
{
    let input_ptr = D::device_ptr(input.buffer);
    let output_ptr = D::device_ptr(output.buffer);
    let Some(plan) =
        plan_axis_scan_launch::<D, Op, T>(device, input, axis, direction, output, width)?
    else {
        return Ok(());
    };
    launch_planned_axis_scan(device, &plan, input_ptr, output_ptr)
}

/// Scan a rank-2 strided operand along `axis`, allocating a C-contiguous
/// output buffer.
///
/// # Errors
///
/// Returns an out-of-range axis, a layout validation failure, the allocation
/// failure, or the backend dispatch failure.
pub fn scan_axis<D, Op, T>(
    device: &D,
    input: StridedView<'_, D::Buffer<T>, 2>,
    axis: usize,
    direction: ScanDirection,
    width: BlockWidth,
) -> Result<D::Buffer<T>>
where
    D: DeviceApi,
    D::CacheKey: From<AxisScanKey>,
    D::Dialect: AxisScanDialect,
    Op: CombineExpr<D::Dialect>,
    T: Pod + IdentityToken<Op, D::Dialect>,
{
    let len = input.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.alloc_zeroed::<T>(0);
    }
    let output_layout = Layout::c_contiguous(input.layout.shape()).map_err(map_layout_err)?;
    let output = device.alloc_uninitialized::<T>(len)?;
    scan_axis_into::<D, Op, T>(
        device,
        input,
        axis,
        direction,
        StridedView::new(&output, &output_layout),
        width,
    )?;
    Ok(output)
}

/// Generate the four direction/operator convenience pairs. Each expands to an
/// `_into` form writing a caller-supplied output and an allocating form; both
/// forward to the one generic entry point above with the operator and
/// direction fixed.
macro_rules! scan_convenience {
    ($($into_name:ident, $alloc_name:ident, $op:ty, $direction:expr, $summary:literal;)*) => {
        $(
            #[doc = concat!($summary, " over a rank-2 strided operand along `axis`.")]
            ///
            /// # Errors
            ///
            /// Returns an out-of-range axis, a shape mismatch, an aliased
            /// output, a layout validation failure, or the backend dispatch
            /// failure.
            #[inline]
            pub fn $into_name<D, T>(
                device: &D,
                input: StridedView<'_, D::Buffer<T>, 2>,
                axis: usize,
                output: StridedView<'_, D::Buffer<T>, 2>,
                width: BlockWidth,
            ) -> Result<()>
            where
                D: DeviceApi,
                D::CacheKey: From<AxisScanKey>,
                D::Dialect: AxisScanDialect,
                $op: CombineExpr<D::Dialect>,
                T: Pod + IdentityToken<$op, D::Dialect>,
            {
                scan_axis_into::<D, $op, T>(device, input, axis, $direction, output, width)
            }

            #[doc = concat!($summary, " over a rank-2 strided operand, allocating a C-contiguous output buffer.")]
            ///
            /// # Errors
            ///
            /// Returns an out-of-range axis, a layout validation failure, the
            /// allocation failure, or the backend dispatch failure.
            #[inline]
            pub fn $alloc_name<D, T>(
                device: &D,
                input: StridedView<'_, D::Buffer<T>, 2>,
                axis: usize,
                width: BlockWidth,
            ) -> Result<D::Buffer<T>>
            where
                D: DeviceApi,
                D::CacheKey: From<AxisScanKey>,
                D::Dialect: AxisScanDialect,
                $op: CombineExpr<D::Dialect>,
                T: Pod + IdentityToken<$op, D::Dialect>,
            {
                scan_axis::<D, $op, T>(device, input, axis, $direction, width)
            }
        )*
    };
}

scan_convenience! {
    cumsum_into, cumsum, CumSumOp, ScanDirection::Forward, "Forward cumulative sum";
    suffix_sum_into, suffix_sum, CumSumOp, ScanDirection::Reverse, "Reverse cumulative sum";
    cumprod_into, cumprod, CumProdOp, ScanDirection::Forward, "Forward cumulative product";
    suffix_prod_into, suffix_prod, CumProdOp, ScanDirection::Reverse, "Reverse cumulative product";
}
