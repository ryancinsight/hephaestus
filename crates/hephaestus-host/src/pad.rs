//! Leto as a [`hephaestus_core::PadOps`] implementor: the CPU reference pair
//! (ADR 0046) for the device-neutral padding seam.
//!
//! Delegates straight to [`leto::pad`] rather than re-deriving the pad
//! geometry: the host substrate is the correctness/conformance reference
//! (crate docs), so its own contract clause and every accelerator's
//! differential test compare against the exact function that defines the
//! semantics, not a second implementation of them.

use eunomia::Pod;
use hephaestus_core::{PadOps, PadWidth, Result, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Leto-backed padding for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostPadOps;

impl<T> PadOps<HostDevice, T> for HostPadOps
where
    T: Pod + Clone,
{
    fn pad_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        width: PadWidth<N>,
        fill: T,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_pad_shape(input.layout.shape(), width, output.layout.shape())?;

        let cells = input.buffer.read();
        let view = ArrayView::<T, N>::try_new(*input.layout, &cells).map_err(map_leto_error)?;
        let padded = leto::pad(&view, width, fill).map_err(map_leto_error)?;
        drop(cells);

        let out_layout = output.layout;
        let mut out_cells = output.buffer.write();
        for (index, value) in padded.indexed_iter() {
            let offset = out_layout.offset_of(index).map_err(map_leto_error)?;
            out_cells[offset] = *value;
        }
        Ok(())
    }
}
