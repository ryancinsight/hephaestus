//! Host reference implementor of [`hephaestus_core::AdaptivePoolingOps`].
//!
//! Walks the two-dimensional index space directly, reducing each output
//! cell's `[start, end)` window (computed by
//! `hephaestus_core::adaptive_window`) with a plain loop — the reference
//! substrate favors an obviously correct implementation over a fast one.

use eunomia::Pod;
use hephaestus_core::{
    AdaptivePoolingMode, AdaptivePoolingOps, Result, StridedView, adaptive_window,
    validate_adaptive_pooling_shape,
};
use leto_ops::Scalar;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Host-backed adaptive average/maximum pooling for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostAdaptivePoolingOps;

impl<T> AdaptivePoolingOps<HostDevice, T> for HostAdaptivePoolingOps
where
    T: Pod + Scalar,
{
    fn adaptive_pool_axis_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, 2>,
        axis: usize,
        mode: AdaptivePoolingMode,
        output: StridedView<'_, HostBuffer<T>, 2>,
    ) -> Result<()> {
        let (lanes, in_len, out_len) =
            validate_adaptive_pooling_shape(axis, input.layout.shape(), output.layout.shape())?;
        let other = 1 - axis;

        let in_cells = input.buffer.read();
        let mut out_cells = output.buffer.write();
        for lane in 0..lanes {
            let mut in_coord = [0usize; 2];
            in_coord[other] = lane;
            let mut out_coord = [0usize; 2];
            out_coord[other] = lane;

            for out_idx in 0..out_len {
                let (start, end) = adaptive_window(out_idx, out_len, in_len);

                let value = match mode {
                    AdaptivePoolingMode::Average => {
                        let mut sum = T::ZERO;
                        for i in start..end {
                            in_coord[axis] = i;
                            let offset =
                                input.layout.offset_of(in_coord).map_err(map_leto_error)?;
                            sum += in_cells[offset];
                        }
                        sum / T::from_usize(end - start)
                    }
                    AdaptivePoolingMode::Maximum => {
                        in_coord[axis] = start;
                        let first_offset =
                            input.layout.offset_of(in_coord).map_err(map_leto_error)?;
                        let mut best = in_cells[first_offset];
                        for i in (start + 1)..end {
                            in_coord[axis] = i;
                            let offset =
                                input.layout.offset_of(in_coord).map_err(map_leto_error)?;
                            let candidate = in_cells[offset];
                            if candidate > best {
                                best = candidate;
                            }
                        }
                        best
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
