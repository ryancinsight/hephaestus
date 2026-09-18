//! Shared plumbing for the host's value-semantic combine dispatch (ADR 0061).
//!
//! [`crate::reduction`] and [`crate::scan`] all resolve their combining
//! operator through [`hephaestus_core::CombineExpr::value`] rather than
//! rendering a kernel expression, because the host executes operators
//! instead of compiling them. This module holds the two pieces that
//! behaviour needs in every implementor: capturing the resolved value
//! function as a plain `fn` pointer at prepare time so the erased `Prepared`
//! associated types never need to name the operator type again, and the
//! offset arithmetic the packed `AxisReductionMeta`/`AxisScanMeta` launch
//! metadata shares with the WGSL/CUDA kernels.

use eunomia::NumericElement;
use hephaestus_core::{CombineExpr, HephaestusError, Host};

/// Resolve `Op`'s host value function once, at prepare time, so `Prepared`
/// forms carry a plain function pointer instead of the erased `Op` type
/// (ADR 0061 Decision 3).
pub(crate) fn combine_fn<Op, T>() -> fn(T, T) -> Option<T>
where
    Op: CombineExpr<Host>,
    T: NumericElement,
{
    <Op as CombineExpr<Host>>::value::<T>
}

/// The typed dispatch failure for an operator with no host value function.
///
/// `Op: CombineExpr<Host>` alone does not imply `CombineValue`: a downstream
/// type may implement the source trait for `Host` without the value trait
/// (ADR 0061 Decision 3), and `value` then reports `None`. This is where the
/// host turns that into a diagnosable error naming the operator, never a
/// silent identity or no-op.
pub(crate) fn unsupported_operator(op_name: &str) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("operator {op_name} has no host value function (ADR 0061)"),
    }
}

/// Physical element offset into a rank-2 buffer at `(row, col)`.
///
/// Matches the formula the `#[repr(C)]` `AxisReductionMeta`/`AxisScanMeta`
/// launch metadata shares with the WGSL/CUDA kernels:
/// `offset + row * strides[0] + col * strides[1]`. The arithmetic runs in
/// `i64` so a strided (e.g. transposed or reversed) view's negative strides
/// are combined without wrapping before the final validated cast to `usize`.
pub(crate) fn offset_2d(base: u32, strides: [i32; 2], row: usize, col: usize) -> usize {
    let row = i64::try_from(row).expect("invariant: axis index fits i64");
    let col = i64::try_from(col).expect("invariant: axis index fits i64");
    let offset = i64::from(base) + row * i64::from(strides[0]) + col * i64::from(strides[1]);
    usize::try_from(offset).expect("invariant: validated layout offset stays in-bounds")
}

/// The length of `axis` in a rank-2 shape, or the same out-of-range
/// rejection [`hephaestus_core::plan_axis_reduction`] gives for a bad axis.
pub(crate) fn axis_len(shape: [usize; 2], axis: usize) -> hephaestus_core::Result<usize> {
    shape
        .get(axis)
        .copied()
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        })
}
