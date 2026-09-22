//! The prepared scalar-operand dispatch.

use eunomia::Pod;
use hephaestus_core::{HephaestusError, Result, StridedView};
use leto::{ArrayView, ArrayViewMut, Layout};

use super::view::{broadcast_operand, map_layout_err, validate_elementwise_output, write_next};
use crate::HostBuffer;

/// A broadcast-scalar dispatch bound to its input/output views, the captured
/// scalar, and a resolved value function.
pub struct HostPreparedScalar<'op, T, const N: usize> {
    input: StridedView<'op, HostBuffer<T>, N>,
    output: StridedView<'op, HostBuffer<T>, N>,
    input_layout: Layout<N>,
    scalar: T,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
}

pub(super) fn prepare_scalar<'op, T, const N: usize>(
    input: StridedView<'op, HostBuffer<T>, N>,
    scalar: T,
    output: StridedView<'op, HostBuffer<T>, N>,
    apply: fn(T, T) -> Option<T>,
    op_name: &'static str,
) -> Result<HostPreparedScalar<'op, T, N>>
where
    T: Pod,
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
    Ok(HostPreparedScalar {
        input,
        output,
        input_layout,
        scalar,
        apply,
        op_name,
    })
}

pub(super) fn dispatch_scalar<T: Copy, const N: usize>(
    prepared: &HostPreparedScalar<'_, T, N>,
) -> Result<()> {
    let input_cells = prepared.input.buffer.read();
    let in_view =
        ArrayView::try_new(prepared.input_layout, &input_cells).map_err(map_layout_err)?;
    let mut output_cells = prepared.output.buffer.write();
    let out_view = ArrayViewMut::try_new(*prepared.output.layout, &mut output_cells)
        .map_err(map_layout_err)?;
    let mut out_iter = out_view.try_iter_mut().map_err(map_layout_err)?;
    for &x in &in_view {
        write_next(
            &mut out_iter,
            (prepared.apply)(x, prepared.scalar),
            prepared.op_name,
        )?;
    }
    Ok(())
}
