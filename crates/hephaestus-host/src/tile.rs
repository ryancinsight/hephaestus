//! Direct-loop n-D tiling for the host reference device.
//!
//! There is no `leto::tile` to delegate to, so this implementation walks
//! the strided output through [`leto::ArrayView`] directly: for every
//! logical index of the output it takes each lane modulo the input
//! extent and copies the addressed input cell. The shared conformance
//! clauses pin these value semantics with hand-computed fixtures every
//! accelerator's differential test must match.

use eunomia::Pod;
use hephaestus_core::{Result, StridedView, TileOps, TileReps};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop tiling for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostTileOps;

impl<T> TileOps<HostDevice, T> for HostTileOps
where
    T: Pod + Copy,
{
    fn tile_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        reps: TileReps<N>,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_tile_shape(input.layout.shape(), reps, output.layout.shape())?;
        if output.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "tile output buffer must not alias the input buffer".to_string(),
            });
        }

        let in_shape = input.layout.shape();
        let in_cells = input.buffer.read();
        let in_view =
            ArrayView::<T, N>::try_new(*input.layout, &in_cells).map_err(map_leto_error)?;

        let mut out_cells = output.buffer.write();
        // Walk the output by logical index: the output shape, not the
        // input shape, defines the iteration domain. A zero extent means
        // zero cells, so no modulo by zero can execute below: any live
        // lane has a nonzero input extent, because `out = in * reps` with
        // `out > 0` proves `in > 0`.
        let out_shape = output.layout.shape();
        if out_shape.contains(&0) {
            return Ok(());
        }
        let mut odometer = [0usize; N];
        loop {
            let mut src = odometer;
            for (d, lane) in src.iter_mut().enumerate() {
                *lane %= in_shape[d];
            }
            let offset = output.layout.offset_of(odometer).map_err(map_leto_error)?;
            out_cells[offset] = in_view[src];

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
