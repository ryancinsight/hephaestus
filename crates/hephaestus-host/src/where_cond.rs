//! Direct-loop n-D conditional selection for the host reference device.
//!
//! There is no `leto::where_cond` to delegate to, so this implementation
//! walks the strided views through [`leto::ArrayView`] directly: for every
//! logical index it tests the condition cell against zero and copies the
//! `on_true` or `on_false` cell. The shared conformance clauses pin these
//! value semantics with hand-computed fixtures every accelerator's
//! differential test must match.

use eunomia::{NumericElement, Pod};
use hephaestus_core::{Result, StridedView, WhereOps};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop conditional selection for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostWhereOps;

impl<T> WhereOps<HostDevice, T> for HostWhereOps
where
    T: Pod + NumericElement,
{
    type Dialect = hephaestus_core::Host;

    fn where_into<const N: usize>(
        &self,
        _device: &HostDevice,
        cond: StridedView<'_, HostBuffer<T>, N>,
        on_true: StridedView<'_, HostBuffer<T>, N>,
        on_false: StridedView<'_, HostBuffer<T>, N>,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_where_shape(
            cond.layout.shape(),
            on_true.layout.shape(),
            on_false.layout.shape(),
            output.layout.shape(),
        )?;
        if output.buffer.aliases(cond.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "where output buffer must not alias the cond buffer".to_string(),
            });
        }
        if output.buffer.aliases(on_true.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "where output buffer must not alias the on_true buffer".to_string(),
            });
        }
        if output.buffer.aliases(on_false.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "where output buffer must not alias the on_false buffer".to_string(),
            });
        }

        let cond_cells = cond.buffer.read();
        let true_cells = on_true.buffer.read();
        let false_cells = on_false.buffer.read();
        let cond_view =
            ArrayView::<T, N>::try_new(*cond.layout, &cond_cells).map_err(map_leto_error)?;
        let true_view =
            ArrayView::<T, N>::try_new(*on_true.layout, &true_cells).map_err(map_leto_error)?;
        let false_view =
            ArrayView::<T, N>::try_new(*on_false.layout, &false_cells).map_err(map_leto_error)?;

        // The condition test is scalar equality against zero, never a
        // bitwise test: `-0.0 == 0.0` selects on_false while `NaN != 0.0`
        // selects on_true.
        let zero = T::ZERO;
        let mut out_cells = output.buffer.write();
        for (index, cond_cell) in cond_view.indexed_iter() {
            let offset = output.layout.offset_of(index).map_err(map_leto_error)?;
            out_cells[offset] = if *cond_cell == zero {
                false_view[index]
            } else {
                true_view[index]
            };
        }
        Ok(())
    }
}
