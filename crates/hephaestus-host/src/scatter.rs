//! Direct-loop n-D scatter-add for the host reference device.
//!
//! There is no `leto::scatter_add` to delegate to, so this implementation
//! copies the strided input into the output and then accumulates through
//! [`leto::ArrayView`] directly: for every source cell in ascending
//! row-major order it bounds-checks the `u32` index value, substitutes it
//! into the `dim` lane, and adds the source cell into the addressed output
//! cell. The shared conformance clauses pin these value semantics with
//! hand-computed fixtures every accelerator's differential test must match.

use std::ops::AddAssign;

use eunomia::Pod;
use hephaestus_core::{Result, ScatterOps, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop scatter-add for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostScatterOps;

impl<T> ScatterOps<HostDevice, T> for HostScatterOps
where
    T: Pod + Copy + AddAssign,
{
    fn scatter_add_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        dim: usize,
        index: StridedView<'_, HostBuffer<u32>, N>,
        src: StridedView<'_, HostBuffer<T>, N>,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_scatter_shape(
            input.layout.shape(),
            dim,
            index.layout.shape(),
            src.layout.shape(),
            output.layout.shape(),
        )?;
        if output.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "scatter output buffer must not alias the input buffer".to_string(),
            });
        }
        if output.buffer.aliases(src.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "scatter output buffer must not alias the source buffer".to_string(),
            });
        }

        let dim_extent = input.layout.shape()[dim];
        let in_cells = input.buffer.read();
        let idx_cells = index.buffer.read();
        let src_cells = src.buffer.read();
        let in_view =
            ArrayView::<T, N>::try_new(*input.layout, &in_cells).map_err(map_leto_error)?;
        let idx_view =
            ArrayView::<u32, N>::try_new(*index.layout, &idx_cells).map_err(map_leto_error)?;
        let src_view =
            ArrayView::<T, N>::try_new(*src.layout, &src_cells).map_err(map_leto_error)?;

        let mut out_cells = output.buffer.write();
        for (index, value) in in_view.indexed_iter() {
            let offset = output.layout.offset_of(index).map_err(map_leto_error)?;
            out_cells[offset] = *value;
        }
        // `indexed_iter` runs in logical row-major order, which is the
        // ascending source order the seam's accumulation contract names.
        for (src_index, raw) in idx_view.indexed_iter() {
            let scatter = usize::try_from(*raw).map_err(|_| {
                hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!("scatter index at {src_index:?} exceeds the usize range"),
                }
            })?;
            if scatter >= dim_extent {
                return Err(hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!(
                        "scatter index {scatter} at {src_index:?} is out of range for dim {dim} extent {dim_extent}",
                    ),
                });
            }
            let mut out_index = src_index;
            out_index[dim] = scatter;
            let offset = output.layout.offset_of(out_index).map_err(map_leto_error)?;
            out_cells[offset] += src_view[src_index];
        }
        Ok(())
    }
}
