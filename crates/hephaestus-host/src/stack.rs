//! Leto as a [`hephaestus_core::StackOps`] implementor: the CPU reference pair
//! (ADR 0046) for the device-neutral stacking seam.
//!
//! Delegates straight to [`leto::stack`] rather than re-deriving the stack
//! geometry: the host substrate is the correctness/conformance reference
//! (crate docs), so its own contract clause and every accelerator's
//! differential test compare against the exact function that defines the
//! semantics, not a second implementation of them.

use eunomia::Pod;
use hephaestus_core::{Result, StackOps, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Leto-backed stacking for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostStackOps;

impl<T> StackOps<HostDevice, T> for HostStackOps
where
    T: Pod + Clone,
{
    fn stack_into<const N: usize, const M: usize>(
        &self,
        _device: &HostDevice,
        inputs: &[StridedView<'_, HostBuffer<T>, N>],
        axis: usize,
        output: StridedView<'_, HostBuffer<T>, M>,
    ) -> Result<()>
    where
        leto::RankMarker<N>: leto::InsertAxis<N, LargerShape = [usize; M]>,
    {
        let shapes: Vec<[usize; N]> = inputs.iter().map(|view| view.layout.shape()).collect();
        hephaestus_core::validate_stack_shape(&shapes, axis, output.layout.shape())?;
        if inputs.iter().any(|view| output.buffer.aliases(view.buffer)) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "stack output buffer must not alias any input buffer".to_string(),
            });
        }

        // Hold every input's read guard alive together: the views borrow the
        // guards, so the two collections stay separate bindings and the
        // guards are never mutated while borrowed.
        let guards: Vec<_> = inputs.iter().map(|view| view.buffer.read()).collect();
        let mut views = Vec::with_capacity(inputs.len());
        for (view, cells) in inputs.iter().zip(guards.iter()) {
            views.push(ArrayView::<T, N>::try_new(*view.layout, cells).map_err(map_leto_error)?);
        }
        let stacked = leto::stack::<T, N, M>(&views, axis).map_err(map_leto_error)?;
        drop(guards);

        let out_layout = output.layout;
        let mut out_cells = output.buffer.write();
        for (index, value) in stacked.indexed_iter() {
            let offset = out_layout.offset_of(index).map_err(map_leto_error)?;
            out_cells[offset] = *value;
        }
        Ok(())
    }
}
