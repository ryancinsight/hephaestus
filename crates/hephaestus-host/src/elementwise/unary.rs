//! The prepared unary dispatch and its preparation and application.

use hephaestus_core::{HephaestusError, Host, Result, StridedView, UnaryExpr};
use leto::{ArrayView, ArrayViewMut, Layout};

use super::dispatch::ElementwiseDispatch;
use super::view::{broadcast_operand, map_layout_err, validate_elementwise_output, write_next};
use crate::HostBuffer;

/// A unary dispatch bound to its input/output views and a resolved value
/// function.
pub struct HostPreparedUnary<'op, T, const N: usize> {
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    input_layout: Layout<N>,
    apply: fn(T) -> Option<T>,
    op_name: &'static str,
}

pub(super) fn prepare_unary<'op, Op, T, const N: usize>(
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
) -> Result<HostPreparedUnary<'op, T, N>>
where
    Op: UnaryExpr<Host>,
    T: ElementwiseDispatch,
{
    if output.buffer.aliases(input.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let input_layout = broadcast_operand(
        input.layout,
        output.layout.shape(),
        input.buffer.read().len(),
    )?;
    validate_elementwise_output(output.buffer, output.layout)?;
    Ok(HostPreparedUnary {
        input,
        output,
        input_layout,
        apply: T::unary_fn::<Op>(),
        op_name: core::any::type_name::<Op>(),
    })
}

pub(super) fn dispatch_unary<T: Copy, const N: usize>(
    prepared: &HostPreparedUnary<'_, T, N>,
) -> Result<()> {
    let input_cells = prepared.input.buffer.read();
    let in_view =
        ArrayView::try_new(prepared.input_layout, &input_cells).map_err(map_layout_err)?;
    let mut output_cells = prepared.output.buffer.write();
    let out_view = ArrayViewMut::try_new(*prepared.output.layout, &mut output_cells)
        .map_err(map_layout_err)?;
    let mut out_iter = out_view.try_iter_mut().map_err(map_layout_err)?;
    for &x in &in_view {
        write_next(&mut out_iter, (prepared.apply)(x), prepared.op_name)?;
    }
    Ok(())
}
