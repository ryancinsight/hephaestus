use eunomia::Pod;
use leto::WindowParameters;

use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;
use crate::domain::window::{
    WindowPlan, invalid, max_offset, plan_window, validate_readonly, validate_writable,
};

use super::{SlidingWindowFoldOperands, SlidingWindowUnfoldOperands};

/// Validated unfold/fold geometry and backend address bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlidingWindowPlan<const S: usize> {
    /// Shared spatial-window geometry.
    pub geometry: WindowPlan<S>,
}

impl<const S: usize> SlidingWindowPlan<S> {
    /// Validate values narrowed by a backend address calculation.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error when the geometry exceeds the
    /// backend's address representation.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        self.geometry.validate_address_limit(max_inclusive)
    }
}

/// Validate an unfold pass before backend preparation.
pub fn plan_sliding_window_unfold<T, B, const R: usize, const S: usize>(
    operands: &SlidingWindowUnfoldOperands<'_, B, R>,
    parameters: WindowParameters<S>,
    illegal_aliasing: bool,
) -> Result<SlidingWindowPlan<S>>
where
    T: Pod,
    B: DeviceBuffer<T>,
{
    let mut geometry =
        plan_window::<T, B, R, S>(operands.input.layout, operands.input.buffer, parameters)?;
    validate_writable(
        operands.output.layout,
        operands.output.buffer,
        "unfold output",
    )?;
    let channel_columns = geometry
        .channels
        .checked_mul(geometry.kernel_volume)
        .ok_or_else(|| invalid("unfold channel column count overflows"))?;
    let expected = [geometry.batch, channel_columns, geometry.output_locations];
    if operands.output.layout.shape() != expected {
        return Err(invalid(format!(
            "unfold output shape {:?} must equal {expected:?}",
            operands.output.layout.shape()
        )));
    }
    reject_aliasing(illegal_aliasing)?;
    geometry.max_physical_offset = geometry
        .max_physical_offset
        .max(max_offset(operands.output.layout)?);
    Ok(SlidingWindowPlan { geometry })
}

/// Validate a fold pass before backend preparation.
pub fn plan_sliding_window_fold<T, B, const R: usize, const S: usize>(
    operands: &SlidingWindowFoldOperands<'_, B, R>,
    output_spatial_shape: [usize; S],
    parameters: WindowParameters<S>,
    illegal_aliasing: bool,
) -> Result<SlidingWindowPlan<S>>
where
    T: Pod,
    B: DeviceBuffer<T>,
{
    let mut geometry =
        plan_window::<T, B, R, S>(operands.output.layout, operands.output.buffer, parameters)?;
    validate_readonly(operands.input.layout, operands.input.buffer)?;
    validate_writable(
        operands.output.layout,
        operands.output.buffer,
        "fold output",
    )?;
    if geometry.input_spatial != output_spatial_shape {
        return Err(invalid(format!(
            "fold output spatial shape {:?} must equal {output_spatial_shape:?}",
            geometry.input_spatial
        )));
    }
    let channel_columns = geometry
        .channels
        .checked_mul(geometry.kernel_volume)
        .ok_or_else(|| invalid("fold channel column count overflows"))?;
    let expected = [geometry.batch, channel_columns, geometry.output_locations];
    if operands.input.layout.shape() != expected {
        return Err(invalid(format!(
            "fold input shape {:?} must equal {expected:?}",
            operands.input.layout.shape()
        )));
    }
    reject_aliasing(illegal_aliasing)?;
    geometry.max_physical_offset = geometry
        .max_physical_offset
        .max(max_offset(operands.input.layout)?);
    Ok(SlidingWindowPlan { geometry })
}

fn reject_aliasing(illegal_aliasing: bool) -> Result<()> {
    if illegal_aliasing {
        return Err(invalid(
            "unfold/fold writable buffers must not alias readable operands",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceBuffer, SlidingWindowUnfoldOperands, StridedView};
    use themis::MemoryTier;

    struct Buffer(usize);

    impl DeviceBuffer<f32> for Buffer {
        fn len(&self) -> usize {
            self.0
        }

        fn tier(&self) -> MemoryTier {
            MemoryTier::Dram
        }
    }

    #[test]
    fn validates_channel_major_unfold_shape() {
        let input_layout = leto::Layout::c_contiguous([1, 2, 4]).expect("input layout");
        let output_layout = leto::Layout::c_contiguous([1, 4, 3]).expect("output layout");
        let input = Buffer(8);
        let output = Buffer(12);
        let plan = plan_sliding_window_unfold::<f32, _, 3, 1>(
            &SlidingWindowUnfoldOperands {
                input: StridedView::new(&input, &input_layout),
                output: StridedView::new(&output, &output_layout),
            },
            WindowParameters::new([2], [1], [0], [1]).expect("valid window parameters"),
            false,
        )
        .expect("valid unfold plan");

        assert_eq!(plan.geometry.output_locations, 3);
        assert_eq!(plan.geometry.kernel_volume, 2);
    }
}
