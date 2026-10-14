//! Leto as a [`hephaestus_core::SplitOps`] implementor: the CPU reference pair
//! (ADR 0046) for the device-neutral splitting seam.
//!
//! Delegates straight to [`leto::split`] rather than re-deriving the split
//! geometry: the host substrate is the correctness/conformance reference
//! (crate docs), so its own contract clause and every accelerator's
//! differential test compare against the exact function that defines the
//! semantics, not a second implementation of them.

use eunomia::Pod;
use hephaestus_core::{Result, SplitOps, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Leto-backed splitting for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostSplitOps;

impl<T> SplitOps<HostDevice, T> for HostSplitOps
where
    T: Pod + Clone,
{
    fn split_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        axis: usize,
        sizes: &[usize],
        outputs: &[StridedView<'_, HostBuffer<T>, N>],
    ) -> Result<()> {
        let shapes: Vec<[usize; N]> = outputs.iter().map(|view| view.layout.shape()).collect();
        hephaestus_core::validate_split_shape(input.layout.shape(), axis, sizes, &shapes)?;
        if outputs.iter().any(|view| view.buffer.aliases(input.buffer)) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "split output buffer must not alias the input buffer".to_string(),
            });
        }
        for (index, first) in outputs.iter().enumerate() {
            if outputs[index + 1..]
                .iter()
                .any(|second| first.buffer.aliases(second.buffer))
            {
                return Err(hephaestus_core::HephaestusError::DispatchFailed {
                    message: "split output buffers must not alias each other".to_string(),
                });
            }
        }

        // The segments borrow the input's read guard, so it stays alive
        // through the copy loop below; the outputs are disjoint buffers by
        // the alias checks above, so their write guards never conflict.
        let guard = input.buffer.read();
        let view = ArrayView::<T, N>::try_new(*input.layout, &guard).map_err(map_leto_error)?;
        let segments = leto::split(&view, axis, sizes).map_err(map_leto_error)?;

        for (segment, output) in segments.iter().zip(outputs.iter()) {
            let mut out_cells = output.buffer.write();
            for (index, value) in segment.indexed_iter() {
                let offset = output.layout.offset_of(index).map_err(map_leto_error)?;
                out_cells[offset] = *value;
            }
        }
        Ok(())
    }
}
