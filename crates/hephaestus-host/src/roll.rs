//! Host reference implementor of [`hephaestus_core::RollOps`].
//!
//! Walks the two-dimensional index space directly, reading each
//! destination's source position from [`hephaestus_core::roll_source_index`]
//! — the reference substrate favors an obviously correct implementation
//! over a fast one.

use eunomia::Pod;
use hephaestus_core::{
    DeviceBuffer, HephaestusError, Result, RollOps, StridedView, roll_source_index,
    validate_roll_shape,
};

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

        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "roll output buffer must not alias input buffer".to_string(),
            });
        }
        input
            .layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_leto_error)?;
        output
            .layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_leto_error)?;
        if !output.layout.is_injective().map_err(map_leto_error)? {
            return Err(HephaestusError::DispatchFailed {
                message: "roll output layout must be non-overlapping".to_string(),
            });
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use hephaestus_core::ComputeDevice;
    use leto::Layout;

    #[test]
    fn rejects_aliased_output_before_locking() {
        let device = HostDevice::new();
        let buffer = device.upload(&[1_i32, 2, 3, 4]).expect("upload");
        let layout = Layout::c_contiguous([2, 2]).expect("layout");
        let error = HostRollOps
            .roll_axis_into(
                &device,
                StridedView::new(&buffer, &layout),
                1,
                1,
                StridedView::new(&buffer, &layout),
            )
            .expect_err("aliased roll output");
        assert!(matches!(error, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn rejects_layout_outside_storage() {
        let device = HostDevice::new();
        let input = device.upload(&[1_i32]).expect("upload");
        let output = device.alloc_zeroed::<i32>(4).expect("output");
        let input_layout = Layout::c_contiguous([2, 2]).expect("layout");
        let output_layout = Layout::c_contiguous([2, 2]).expect("layout");
        let error = HostRollOps
            .roll_axis_into(
                &device,
                StridedView::new(&input, &input_layout),
                1,
                1,
                StridedView::new(&output, &output_layout),
            )
            .expect_err("short input storage");
        assert!(matches!(error, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn rejects_non_injective_output_layout() {
        let device = HostDevice::new();
        let input = device.upload(&[1_i32, 2, 3, 4]).expect("upload");
        let output = device.alloc_zeroed::<i32>(2).expect("output");
        let input_layout = Layout::c_contiguous([2, 2]).expect("layout");
        let output_layout = Layout::try_new([2, 2], [0, 1], 0).expect("broadcast layout");
        let error = HostRollOps
            .roll_axis_into(
                &device,
                StridedView::new(&input, &input_layout),
                1,
                1,
                StridedView::new(&output, &output_layout),
            )
            .expect_err("broadcast output layout");
        assert!(matches!(error, HephaestusError::DispatchFailed { .. }));
    }
}
