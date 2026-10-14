//! Direct-loop n-D circular shift for the host reference device.
//!
//! There is no `leto::roll` to delegate to, so this implementation folds
//! the `(shift, dim)` pairs into one effective per-axis displacement and
//! reads the strided input through [`leto::ArrayView`] directly: for every
//! logical index of the output it moves each rolled lane back by its
//! effective shift modulo the extent and copies the addressed input cell.
//! Rolls along distinct dims commute and rolls along one dim add, so the
//! fold matches sequential application exactly, including duplicate dims.
//! The shared conformance clauses pin these value semantics with
//! hand-computed fixtures every accelerator's differential test must
//! match.

use eunomia::Pod;
use hephaestus_core::{Result, RollOps, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop circular shift for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostRollOps;

impl<T> RollOps<HostDevice, T> for HostRollOps
where
    T: Pod + Copy,
{
    fn roll_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        shifts: &[isize],
        dims: &[usize],
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_roll_shape(
            input.layout.shape(),
            shifts,
            dims,
            output.layout.shape(),
        )?;
        if output.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "roll output buffer must not alias the input buffer".to_string(),
            });
        }

        let in_shape = input.layout.shape();
        // Fold every pair into one effective per-axis shift in i128: the
        // sum cannot overflow for any realistic pair list, and reducing
        // once per axis keeps the per-cell loop to a single modulo.
        let mut totals = [0i128; N];
        for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
            totals[dim] += shift as i128;
        }
        let mut back = [0usize; N];
        for (d, (&total, extent)) in totals.iter().zip(in_shape.iter()).enumerate() {
            if *extent > 0 {
                // The reduced shift is below the extent, hence below
                // `usize::MAX`; the fallible conversion keeps the function
                // total regardless.
                let reduced = total.rem_euclid(*extent as i128);
                back[d] = usize::try_from(reduced).map_err(|_| {
                    hephaestus_core::HephaestusError::InvalidConfiguration {
                        message: format!("roll back-shift {reduced} exceeds the usize range"),
                    }
                })?;
            }
        }

        let in_cells = input.buffer.read();
        let in_view =
            ArrayView::<T, N>::try_new(*input.layout, &in_cells).map_err(map_leto_error)?;

        let mut out_cells = output.buffer.write();
        // Input and output share shape, so the input's index domain is the
        // output's. A zero extent yields no indices, so no modulo by zero
        // can execute below.
        for (index, _) in in_view.indexed_iter() {
            let mut src = index;
            for (d, &b) in back.iter().enumerate() {
                let n = in_shape[d];
                src[d] = (index[d] + n - b) % n;
            }
            let offset = output.layout.offset_of(index).map_err(map_leto_error)?;
            out_cells[offset] = in_view[src];
        }
        Ok(())
    }
}
