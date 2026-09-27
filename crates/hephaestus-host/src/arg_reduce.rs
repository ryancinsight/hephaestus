//! Host reference implementor of [`hephaestus_core::ArgReduceOps`].
//!
//! `leto::argmax`/`argmin` drop the reduced axis (rank `N` -> `N - 1`), but
//! this seam keeps it at length one (`hephaestus_core::AxisReductionOps`'s
//! convention, so the result stays broadcastable against the input) — a rank
//! leto's own functions do not return. Rather than reshape leto's output
//! back in, this walks the two-dimensional index space directly, using
//! leto's own strict-first-occurrence-wins tie-break rule (`candidate > best`
//! / `candidate < best`), so host and device agree bit-for-bit.

use eunomia::Pod;
use hephaestus_core::{
    ArgReduceOps, HephaestusError, Result, StridedView, validate_arg_reduce_shape,
};

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Host-backed argmax/argmin for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostArgReduceOps;

/// Scan `input`'s lanes along `axis`, writing the index selected by
/// `is_better(current_best, candidate)` (strict: the first occurrence wins on
/// ties) into `output`.
fn arg_reduce_into<T: Pod + PartialOrd>(
    input: StridedView<'_, HostBuffer<T>, 2>,
    axis: usize,
    output: StridedView<'_, HostBuffer<u32>, 2>,
    is_better: impl Fn(T, T) -> bool,
) -> Result<()> {
    let (lanes, axis_len) =
        validate_arg_reduce_shape(axis, input.layout.shape(), output.layout.shape())?;
    let other = 1 - axis;

    let in_cells = input.buffer.read();
    let mut out_cells = output.buffer.write();
    for lane in 0..lanes {
        let mut coord = [0usize; 2];
        coord[other] = lane;
        coord[axis] = 0;
        let first_offset = input.layout.offset_of(coord).map_err(map_leto_error)?;
        let mut best_val = in_cells[first_offset];
        let mut best_idx = 0u32;
        for a in 1..axis_len {
            coord[axis] = a;
            let offset = input.layout.offset_of(coord).map_err(map_leto_error)?;
            let candidate = in_cells[offset];
            if is_better(best_val, candidate) {
                best_val = candidate;
                best_idx = u32::try_from(a).map_err(|_| HephaestusError::InvalidConfiguration {
                    message: format!("axis index {a} exceeds u32 range"),
                })?;
            }
        }
        let mut out_coord = [0usize; 2];
        out_coord[other] = lane;
        let out_offset = output.layout.offset_of(out_coord).map_err(map_leto_error)?;
        out_cells[out_offset] = best_idx;
    }
    Ok(())
}

impl<T> ArgReduceOps<HostDevice, T> for HostArgReduceOps
where
    T: Pod + PartialOrd,
{
    fn argmax_axis_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, HostBuffer<u32>, 2>,
    ) -> Result<()> {
        arg_reduce_into(input, axis, output, |best, candidate| candidate > best)
    }

    fn argmin_axis_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, 2>,
        axis: usize,
        output: StridedView<'_, HostBuffer<u32>, 2>,
    ) -> Result<()> {
        arg_reduce_into(input, axis, output, |best, candidate| candidate < best)
    }
}
