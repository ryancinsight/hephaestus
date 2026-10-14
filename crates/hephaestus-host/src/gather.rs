//! Direct-loop n-D gather for the host reference device.
//!
//! There is no `leto::gather` to delegate to, so this implementation reads
//! the strided input through [`leto::ArrayView`] directly: for every logical
//! index of the output it bounds-checks the `u32` index value, substitutes
//! it into the `dim` lane, and copies the addressed input cell. The shared
//! conformance clauses pin these value semantics with hand-computed
//! fixtures every accelerator's differential test must match.

use eunomia::Pod;
use hephaestus_core::{GatherOps, Result, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop gather for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostGatherOps;

impl<T> GatherOps<HostDevice, T> for HostGatherOps
where
    T: Pod + Clone,
{
    fn gather_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, HostBuffer<u32>, N>,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_gather_shape(
            input.layout.shape(),
            dim,
            index.layout.shape(),
            output.layout.shape(),
        )?;
        if output.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "gather output buffer must not alias the input buffer".to_string(),
            });
        }

        let dim_extent = input.layout.shape()[dim];
        let in_cells = input.buffer.read();
        let idx_cells = index.buffer.read();
        let in_view =
            ArrayView::<T, N>::try_new(*input.layout, &in_cells).map_err(map_leto_error)?;
        let idx_view =
            ArrayView::<u32, N>::try_new(*index.layout, &idx_cells).map_err(map_leto_error)?;

        let mut out_cells = output.buffer.write();
        for (out_index, raw) in idx_view.indexed_iter() {
            let gather = usize::try_from(*raw).map_err(|_| {
                hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!("gather index at {out_index:?} exceeds the usize range"),
                }
            })?;
            if gather >= dim_extent {
                return Err(hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!(
                        "gather index {gather} at {out_index:?} is out of range for dim {dim} extent {dim_extent}",
                    ),
                });
            }
            let mut in_index = out_index;
            in_index[dim] = gather;
            let offset = output.layout.offset_of(out_index).map_err(map_leto_error)?;
            out_cells[offset] = in_view[in_index];
        }
        Ok(())
    }
}
