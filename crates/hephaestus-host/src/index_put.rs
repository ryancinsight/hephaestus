//! Direct-loop n-D scatter assignment for the host reference device.
//!
//! There is no `leto::index_put` to delegate to, so this implementation
//! copies the strided base input into the output and then scatters
//! through [`leto::ArrayView`] directly: for every selection `s` in
//! ascending order it bounds-checks the `u32` index value, substitutes it
//! into the `dim` lane, and writes or accumulates the values slice into
//! the addressed output cells. The shared conformance clauses pin these
//! value semantics with hand-computed fixtures every accelerator's
//! differential test must match.

use std::ops::AddAssign;

use eunomia::Pod;
use hephaestus_core::{IndexPutOps, Result, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop scatter assignment for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostIndexPutOps;

impl<T> IndexPutOps<HostDevice, T> for HostIndexPutOps
where
    T: Pod + Copy + AddAssign,
{
    fn index_put_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, HostBuffer<u32>, 1>,
        values: StridedView<'_, HostBuffer<T>, N>,
        accumulate: bool,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_index_put_shape(
            input.layout.shape(),
            dim,
            index.layout.shape()[0],
            values.layout.shape(),
            output.layout.shape(),
        )?;
        if output.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "index_put output buffer must not alias the input buffer".to_string(),
            });
        }
        if output.buffer.aliases(values.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "index_put output buffer must not alias the values buffer".to_string(),
            });
        }

        let dim_extent = input.layout.shape()[dim];
        let index_len = index.layout.shape()[0];
        let in_cells = input.buffer.read();
        let idx_cells = index.buffer.read();
        let vals_cells = values.buffer.read();
        let in_view =
            ArrayView::<T, N>::try_new(*input.layout, &in_cells).map_err(map_leto_error)?;
        let idx_view =
            ArrayView::<u32, 1>::try_new(*index.layout, &idx_cells).map_err(map_leto_error)?;
        let vals_view =
            ArrayView::<T, N>::try_new(*values.layout, &vals_cells).map_err(map_leto_error)?;

        let mut out_cells = output.buffer.write();
        for (index, value) in in_view.indexed_iter() {
            let offset = output.layout.offset_of(index).map_err(map_leto_error)?;
            out_cells[offset] = *value;
        }
        // An empty selection updates vacuously: no cell reads any index,
        // so no out-of-range value can fault.
        if index_len == 0 {
            return Ok(());
        }
        let vals_shape = values.layout.shape();
        if vals_shape.contains(&0) {
            return Ok(());
        }
        // Ascending selection order is the seam's accumulation contract:
        // the outer loop over `s` pins it structurally rather than
        // relying on `indexed_iter` lane order.
        for s in 0..index_len {
            let raw = idx_view[[s]];
            let select = usize::try_from(raw).map_err(|_| {
                hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!("index_put index at [{s}] exceeds the usize range"),
                }
            })?;
            if select >= dim_extent {
                return Err(hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!(
                        "index_put index {select} at [{s}] is out of range for dim {dim} extent {dim_extent}",
                    ),
                });
            }
            let mut odometer = [0usize; N];
            odometer[dim] = s;
            'slice: loop {
                let mut out_index = odometer;
                out_index[dim] = select;
                let offset = output.layout.offset_of(out_index).map_err(map_leto_error)?;
                if accumulate {
                    out_cells[offset] += vals_view[odometer];
                } else {
                    out_cells[offset] = vals_view[odometer];
                }

                // Advance the odometer past every lane but `dim`, which
                // stays pinned at the current selection.
                let mut lane = N;
                loop {
                    if lane == 0 {
                        break 'slice;
                    }
                    lane -= 1;
                    if lane == dim {
                        continue;
                    }
                    odometer[lane] += 1;
                    if odometer[lane] < vals_shape[lane] {
                        break;
                    }
                    odometer[lane] = 0;
                }
            }
        }
        Ok(())
    }
}
