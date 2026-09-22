//! The prepared binary dispatch, including the aliased-operand read path.

use eunomia::Pod;
use hephaestus_core::{Result, StridedView};
use leto::{ArrayView, ArrayViewMut, ElementIterMut, Layout};

use super::view::{broadcast_operand, map_layout_err, validate_elementwise_output, write_next};
use crate::HostBuffer;
use crate::operands::require_disjoint_output;

/// A binary dispatch bound to its operand/output views and a resolved value
/// function. Shared by [`hephaestus_core::ElementwiseOps::PreparedBinary`]
/// and [`hephaestus_core::ElementwiseOps::PreparedTypedBinary`]: both prepare
/// an identical shape
/// (two broadcast operand layouts, one output, one resolved
/// `fn(T, T) -> Option<T>`) and differ only in which source trait resolved
/// that function.
pub struct HostPreparedBinary<'op, T, const N: usize> {
    lhs: StridedView<'op, HostBuffer<T>, N>,
    rhs: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    lhs_layout: Layout<N>,
    rhs_layout: Layout<N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
}

/// Apply `apply` across two same-shape views into `out_iter`, in the shared
/// logical row-major order every leto element iterator walks.
pub(super) fn apply_zipped<T: Copy, const N: usize>(
    lhs_view: &ArrayView<'_, T, N>,
    rhs_view: &ArrayView<'_, T, N>,
    out_iter: &mut ElementIterMut<'_, T, N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &str,
) -> Result<()> {
    for (&l, &r) in lhs_view.iter().zip(rhs_view) {
        write_next(out_iter, apply(l, r), op_name)?;
    }
    Ok(())
}

pub(super) fn prepare_binary<'op, T, const N: usize>(
    lhs: StridedView<'op, HostBuffer<T>, N>,
    rhs: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
) -> Result<HostPreparedBinary<'op, T, N>>
where
    T: Pod,
{
    require_disjoint_output(lhs.buffer, rhs.buffer, output.buffer)?;
    let out_shape = output.layout.shape();
    let lhs_layout = broadcast_operand(lhs.layout, out_shape, lhs.buffer.read().len())?;
    let rhs_layout = broadcast_operand(rhs.layout, out_shape, rhs.buffer.read().len())?;
    validate_elementwise_output(output.buffer, output.layout)?;
    Ok(HostPreparedBinary {
        lhs,
        rhs,
        output,
        lhs_layout,
        rhs_layout,
        apply,
        op_name,
    })
}

pub(super) fn dispatch_binary<T: Copy, const N: usize>(
    prepared: &HostPreparedBinary<'_, T, N>,
) -> Result<()> {
    let mut output_cells = prepared.output.buffer.write();
    let out_view = ArrayViewMut::try_new(*prepared.output.layout, &mut output_cells)
        .map_err(map_layout_err)?;
    let mut out_iter = out_view.try_iter_mut().map_err(map_layout_err)?;

    let lhs_cells = prepared.lhs.buffer.read();
    let lhs_view = ArrayView::try_new(prepared.lhs_layout, &lhs_cells).map_err(map_layout_err)?;

    if prepared.rhs.buffer.aliases(prepared.lhs.buffer) {
        // One RwLock guard serves both operands (`crate::operands`): a
        // second `read()` on the same lock is unsound to rely on.
        let rhs_view =
            ArrayView::try_new(prepared.rhs_layout, &lhs_cells).map_err(map_layout_err)?;
        apply_zipped(
            &lhs_view,
            &rhs_view,
            &mut out_iter,
            prepared.apply,
            prepared.op_name,
        )
    } else {
        let rhs_cells = prepared.rhs.buffer.read();
        let rhs_view =
            ArrayView::try_new(prepared.rhs_layout, &rhs_cells).map_err(map_layout_err)?;
        apply_zipped(
            &lhs_view,
            &rhs_view,
            &mut out_iter,
            prepared.apply,
            prepared.op_name,
        )
    }
}
