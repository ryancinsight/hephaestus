//! Host reference implementor of [`hephaestus_core::InterpolationOps`].
//!
//! Walks the two-dimensional index space directly and computes the mapped
//! source coordinate's fractional weight entirely in `T`'s native precision
//! via `leto_ops::Scalar::from_usize` — no widen-compute-narrow cast (HARD
//! per `integrity`: fake generics; `numerical_discipline`: concrete
//! precision contract).

use eunomia::Pod;
use hephaestus_core::{
    InterpolationMode, InterpolationOps, Result, StridedView, validate_interpolation_shape,
};
use leto_ops::Scalar;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Host-backed rank-2 axis resampling for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostInterpolationOps;

/// Map output index `out_idx` (of `out_len`) onto a source coordinate over
/// `[0, in_len - 1]` under the align-corners convention, returning the lower
/// bracketing index and the fractional weight toward the upper one.
fn source_coordinate<T: Scalar>(out_idx: usize, out_len: usize, in_len: usize) -> (usize, T) {
    if in_len == 1 || out_len == 1 {
        return (0, T::ZERO);
    }
    let numerator = T::from_usize(out_idx) * T::from_usize(in_len - 1);
    let denominator = T::from_usize(out_len - 1);
    let src = numerator / denominator;
    let lower = out_idx * (in_len - 1) / (out_len - 1);
    let lower = lower.min(in_len - 1);
    let frac = src - T::from_usize(lower);
    (lower, frac)
}

impl<T> InterpolationOps<HostDevice, T> for HostInterpolationOps
where
    T: Pod + Scalar,
{
    fn interpolate_axis_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, 2>,
        axis: usize,
        mode: InterpolationMode,
        output: StridedView<'_, HostBuffer<T>, 2>,
    ) -> Result<()> {
        let (lanes, in_len, out_len) =
            validate_interpolation_shape(axis, input.layout.shape(), output.layout.shape())?;
        let other = 1 - axis;

        let in_cells = input.buffer.read();
        let mut out_cells = output.buffer.write();
        for lane in 0..lanes {
            let mut in_coord = [0usize; 2];
            in_coord[other] = lane;
            let mut out_coord = [0usize; 2];
            out_coord[other] = lane;

            for out_idx in 0..out_len {
                let (lower, frac) = source_coordinate::<T>(out_idx, out_len, in_len);

                in_coord[axis] = lower;
                let lower_offset = input.layout.offset_of(in_coord).map_err(map_leto_error)?;
                let lower_val = in_cells[lower_offset];

                let value = match mode {
                    InterpolationMode::Nearest => {
                        // Round-half-down: `frac < 0.5` keeps `lower`,
                        // `frac >= 0.5` advances to `lower + 1` (clamped).
                        if frac < T::from_usize(1) / T::from_usize(2) {
                            lower_val
                        } else {
                            let upper = (lower + 1).min(in_len - 1);
                            in_coord[axis] = upper;
                            let upper_offset =
                                input.layout.offset_of(in_coord).map_err(map_leto_error)?;
                            in_cells[upper_offset]
                        }
                    }
                    InterpolationMode::Linear => {
                        let upper = (lower + 1).min(in_len - 1);
                        in_coord[axis] = upper;
                        let upper_offset =
                            input.layout.offset_of(in_coord).map_err(map_leto_error)?;
                        let upper_val = in_cells[upper_offset];
                        lower_val + (upper_val - lower_val) * frac
                    }
                };

                out_coord[axis] = out_idx;
                let out_offset = output.layout.offset_of(out_coord).map_err(map_leto_error)?;
                out_cells[out_offset] = value;
            }
        }
        Ok(())
    }
}
