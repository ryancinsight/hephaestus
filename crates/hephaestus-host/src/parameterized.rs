//! Runtime-parameter unary elementwise operations for the host reference
//! device (ADR 0061).
//!
//! [`HostParameterizedUnaryOps`] resolves the operator through
//! [`ParameterizedUnaryExpr::value`](hephaestus_core::ParameterizedUnaryExpr::value)
//! rather than rendering a kernel
//! expression, mirroring [`crate::elementwise`]'s unary dispatch: the input
//! broadcasts to the output's shape, the output buffer must not alias the
//! input, and the output layout must be non-overlapping (validated with
//! [`hephaestus_core::validate_parameterized_output`], the exact allocation-
//! bounded proof [`hephaestus_core::ParameterizedUnaryOps`]'s own
//! documentation describes, shared with every other backend).

use hephaestus_core::{
    HephaestusError, Host, ParameterizedUnaryExpr, ParameterizedUnaryOps, Result, StridedView,
    validate_parameterized_output,
};
use leto::{ArrayView, ArrayViewMut};

use crate::elementwise::{broadcast_operand, map_layout_err, write_next};
use crate::{HostBuffer, HostDevice};

/// Runtime-parameter unary elementwise operations for the host reference
/// device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostParameterizedUnaryOps;

impl ParameterizedUnaryOps<HostDevice> for HostParameterizedUnaryOps {
    type Dialect = Host;

    fn parameterized_unary_into<Op, const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<f32>, N>,
        parameters: [f32; 2],
        output: StridedView<'_, HostBuffer<f32>, N>,
    ) -> Result<()>
    where
        Op: ParameterizedUnaryExpr<Self::Dialect>,
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
        let len = validate_parameterized_output(output.layout, output.buffer.read().len())?;
        if len == 0 {
            return Ok(());
        }

        let apply = <Op as ParameterizedUnaryExpr<Host>>::value::<f32>;
        let op_name = core::any::type_name::<Op>();
        let input_cells = input.buffer.read();
        let in_view = ArrayView::try_new(input_layout, &input_cells).map_err(map_layout_err)?;
        let mut output_cells = output.buffer.write();
        let out_view =
            ArrayViewMut::try_new(*output.layout, &mut output_cells).map_err(map_layout_err)?;
        let mut out_iter = out_view.try_iter_mut().map_err(map_layout_err)?;
        for &x in &in_view {
            write_next(
                &mut out_iter,
                apply(x, parameters[0], parameters[1]),
                op_name,
            )?;
        }
        Ok(())
    }
}
