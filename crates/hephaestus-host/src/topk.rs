//! Host reference implementor of [`hephaestus_core::TopKOps`].
//!
//! The same incremental insertion used by the device kernel (fill the first
//! `k` slots in sorted-descending order, then displace the current minimum
//! whenever a later element is strictly greater) rather than a full sort —
//! matching the algorithm, not just the output contract, is what lets the
//! host and device agree on tie-breaking by construction rather than by
//! coincidence.

use eunomia::Pod;
use hephaestus_core::{Result, StridedView, TopKOps, validate_topk_shape};

use crate::HostDevice;
use crate::operands::require_disjoint_output;

/// Host-backed top-k selection for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostTopKOps;

impl<T> TopKOps<HostDevice, T> for HostTopKOps
where
    T: Pod + PartialOrd + Copy,
{
    fn topk_axis_into(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, crate::HostBuffer<T>, 2>,
        axis: usize,
        k: usize,
        values: StridedView<'_, crate::HostBuffer<T>, 2>,
        indices: StridedView<'_, crate::HostBuffer<u32>, 2>,
    ) -> Result<()> {
        require_disjoint_output(input.buffer, input.buffer, values.buffer)?;

        let (lanes, axis_len) = validate_topk_shape(
            axis,
            k,
            input.layout.shape(),
            values.layout.shape(),
            indices.layout.shape(),
        )?;
        let other = 1 - axis;

        let in_cells = input.buffer.read();
        let mut val_cells = values.buffer.write();
        let mut idx_cells = indices.buffer.write();

        for lane in 0..lanes {
            let mut coord = [0usize; 2];
            coord[other] = lane;
            let mut top_vals: Vec<T> = Vec::with_capacity(k);
            let mut top_idxs: Vec<u32> = Vec::with_capacity(k);

            let insert = |top_vals: &mut Vec<T>, top_idxs: &mut Vec<u32>, val: T, idx: u32| {
                let mut pos;
                if top_vals.len() < k {
                    pos = top_vals.len();
                    top_vals.push(val);
                    top_idxs.push(idx);
                } else if val > top_vals[k - 1] {
                    pos = k - 1;
                    top_vals[pos] = val;
                    top_idxs[pos] = idx;
                } else {
                    return;
                }
                while pos > 0 && top_vals[pos - 1] < top_vals[pos] {
                    top_vals.swap(pos - 1, pos);
                    top_idxs.swap(pos - 1, pos);
                    pos -= 1;
                }
            };

            for a in 0..axis_len {
                coord[axis] = a;
                let offset = input
                    .layout
                    .offset_of(coord)
                    .map_err(crate::map_leto_error)?;
                let val = in_cells[offset];
                insert(
                    &mut top_vals,
                    &mut top_idxs,
                    val,
                    u32::try_from(a).expect("axis index fits u32 within a rank-2 shape"),
                );
            }

            for slot in 0..k {
                let mut out_coord = [0usize; 2];
                out_coord[other] = lane;
                out_coord[axis] = slot;
                let val_offset = values
                    .layout
                    .offset_of(out_coord)
                    .map_err(crate::map_leto_error)?;
                let idx_offset = indices
                    .layout
                    .offset_of(out_coord)
                    .map_err(crate::map_leto_error)?;
                val_cells[val_offset] = top_vals[slot];
                idx_cells[idx_offset] = top_idxs[slot];
            }
        }
        Ok(())
    }
}
