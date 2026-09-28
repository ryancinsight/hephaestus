//! Host reference implementor of [`hephaestus_core::RollOps`].
//!
//! Walks the two-dimensional index space directly, reading each
//! destination's source position from [`hephaestus_core::roll_source_index`]
//! — the reference substrate favors an obviously correct implementation
//! over a fast one.

use eunomia::Pod;
use hephaestus_core::{Result, RollOps, StridedView, roll_source_index, validate_roll_shape};

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Host-backed axis roll for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostRollOps;

impl<T> RollOps<HostDevice, T> for HostRollOps
where
    T: Pod,
{
    fn roll_axis_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, 2>,
        axis: usize,
        shift: i64,
        output: StridedView<'_, HostBuffer<T>, 2>,
    ) -> Result<()> {
        let (lanes, axis_len) =
            validate_roll_shape(axis, input.layout.shape(), output.layout.shape())?;
        let other = 1 - axis;

        let in_cells = input.buffer.read();
        let mut out_cells = output.buffer.write();
        for lane in 0..lanes {
            let mut in_coord = [0usize; 2];
            in_coord[other] = lane;
            let mut out_coord = [0usize; 2];
            out_coord[other] = lane;

            for dst_idx in 0..axis_len {
                let src_idx = roll_source_index(dst_idx, shift, axis_len);
                in_coord[axis] = src_idx;
                let in_offset = input.layout.offset_of(in_coord).map_err(map_leto_error)?;
                out_coord[axis] = dst_idx;
                let out_offset = output.layout.offset_of(out_coord).map_err(map_leto_error)?;
                out_cells[out_offset] = in_cells[in_offset];
            }
        }
        Ok(())
    }
}
