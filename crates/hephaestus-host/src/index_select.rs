//! Direct-loop n-D index selection for the host reference device.
//!
//! There is no `leto::index_select` to delegate to, so this implementation
//! reads the strided input through [`leto::ArrayView`] directly: for every
//! logical index of the output it reads the `u32` index value at the
//! index's own `dim` coordinate, bounds-checks it, substitutes it into the
//! `dim` lane, and copies the addressed input cell. The shared conformance
//! clauses pin these value semantics with hand-computed fixtures every
//! accelerator's differential test must match.

use eunomia::Pod;
use hephaestus_core::{IndexSelectOps, Result, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop index selection for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostIndexSelectOps;

impl<T> IndexSelectOps<HostDevice, T> for HostIndexSelectOps
where
    T: Pod + Clone,
{
    fn index_select_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, HostBuffer<u32>, 1>,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_index_select_shape(
            input.layout.shape(),
            dim,
            index.layout.shape()[0],
            output.layout.shape(),
        )?;
        if output.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "index_select output buffer must not alias the input buffer".to_string(),
            });
        }

        let dim_extent = input.layout.shape()[dim];
        let in_cells = input.buffer.read();
        let idx_cells = index.buffer.read();
        let in_view =
            ArrayView::<T, N>::try_new(*input.layout, &in_cells).map_err(map_leto_error)?;
        let idx_view =
            ArrayView::<u32, 1>::try_new(*index.layout, &idx_cells).map_err(map_leto_error)?;

        let mut out_cells = output.buffer.write();
        // Walk the output by logical index: the output shape, not the
        // input shape, defines the iteration domain. A zero extent means
        // zero cells, and no cell reads any index, so no out-of-range
        // value can fault.
        let out_shape = output.layout.shape();
        if out_shape.contains(&0) {
            return Ok(());
        }
        let mut odometer = [0usize; N];
        loop {
            let slot = odometer[dim];
            let raw = idx_view[[slot]];
            let select = usize::try_from(raw).map_err(|_| {
                hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!("index_select index at [{slot}] exceeds the usize range"),
                }
            })?;
            if select >= dim_extent {
                return Err(hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!(
                        "index_select index {select} at [{slot}] is out of range for dim {dim} extent {dim_extent}",
                    ),
                });
            }
            let mut in_index = odometer;
            in_index[dim] = select;
            let offset = output.layout.offset_of(odometer).map_err(map_leto_error)?;
            out_cells[offset] = in_view[in_index];

            // Advance the odometer; stop after the trailing lane wraps.
            let mut lane = N;
            loop {
                if lane == 0 {
                    return Ok(());
                }
                lane -= 1;
                odometer[lane] += 1;
                if odometer[lane] < out_shape[lane] {
                    break;
                }
                odometer[lane] = 0;
            }
        }
    }
}
