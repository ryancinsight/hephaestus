use bytemuck::Pod;
use leto::WindowParameters;

use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::Result;
use crate::domain::window::{
    WindowPlan, invalid, max_offset, plan_window, validate_shape, validate_writable,
};

use super::{PoolingBackwardOperands, PoolingForwardOperands};

/// Validated pooling geometry and backend address bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolingPlan<const S: usize> {
    /// Shared spatial-window geometry.
    pub geometry: WindowPlan<S>,
}

impl<const S: usize> PoolingPlan<S> {
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

/// Validate a pooling forward pass before backend preparation.
pub fn plan_pooling_forward<T, B, const R: usize, const S: usize>(
    operands: &PoolingForwardOperands<'_, B, R>,
    parameters: WindowParameters<S>,
    illegal_aliasing: bool,
) -> Result<PoolingPlan<S>>
where
    T: Pod,
    B: DeviceBuffer<T>,
{
    let mut geometry =
        plan_window::<T, B, R, S>(operands.input.layout, operands.input.buffer, parameters)?;
    validate_writable(operands.output.layout, operands.output.buffer, "output")?;
    validate_shape(
        operands.output.layout.shape(),
        geometry.batch,
        geometry.channels,
        &geometry.output_spatial,
        "output",
    )?;
    reject_aliasing(illegal_aliasing)?;
    geometry.max_physical_offset = geometry
        .max_physical_offset
        .max(max_offset(operands.output.layout)?);
    Ok(PoolingPlan { geometry })
}

/// Validate an additive pooling backward pass before backend preparation.
pub fn plan_pooling_backward<T, B, const R: usize, const S: usize>(
    operands: &PoolingBackwardOperands<'_, B, R>,
    parameters: WindowParameters<S>,
    illegal_aliasing: bool,
) -> Result<PoolingPlan<S>>
where
    T: Pod,
    B: DeviceBuffer<T>,
{
    let mut geometry =
        plan_window::<T, B, R, S>(operands.input.layout, operands.input.buffer, parameters)?;
    validate_shape(
        operands.grad_output.layout.shape(),
        geometry.batch,
        geometry.channels,
        &geometry.output_spatial,
        "gradient output",
    )?;
    validate_shape(
        operands.grad_input.layout.shape(),
        geometry.batch,
        geometry.channels,
        &geometry.input_spatial,
        "gradient input",
    )?;
    validate_writable(
        operands.grad_input.layout,
        operands.grad_input.buffer,
        "gradient input",
    )?;
    operands
        .grad_output
        .layout
        .validate_storage_len(operands.grad_output.buffer.len())
        .map_err(crate::domain::planning::map_layout_err)?;
    reject_aliasing(illegal_aliasing)?;
    geometry.max_physical_offset = geometry
        .max_physical_offset
        .max(max_offset(operands.grad_output.layout)?)
        .max(max_offset(operands.grad_input.layout)?);
    Ok(PoolingPlan { geometry })
}

fn reject_aliasing(illegal_aliasing: bool) -> Result<()> {
    if illegal_aliasing {
        return Err(invalid(
            "pooling writable buffers must not alias readable operands",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceBuffer, StridedView};
    use leto::{Layout, WindowParameters};
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

    fn parameters() -> WindowParameters<2> {
        WindowParameters::new([3, 2], [2, 1], [1, 0], [1, 1]).expect("valid pooling parameters")
    }

    #[test]
    fn derives_batch_channel_and_spatial_geometry() {
        let input_layout = Layout::c_contiguous([2, 3, 5, 4]).expect("input layout");
        let output_layout = Layout::c_contiguous([2, 3, 3, 3]).expect("output layout");
        let input = Buffer(120);
        let output = Buffer(54);
        let plan = plan_pooling_forward::<f32, _, 4, 2>(
            &PoolingForwardOperands {
                input: StridedView::new(&input, &input_layout),
                output: StridedView::new(&output, &output_layout),
            },
            parameters(),
            false,
        )
        .expect("valid pooling plan");

        assert_eq!(plan.geometry.batch, 2);
        assert_eq!(plan.geometry.channels, 3);
        assert_eq!(plan.geometry.input_spatial, [5, 4]);
        assert_eq!(plan.geometry.output_spatial, [3, 3]);
        assert_eq!(plan.geometry.kernel_volume, 6);
        assert_eq!(plan.geometry.output_locations, 9);
    }

    #[test]
    fn rejects_aliasing_before_backend_preparation() {
        let input_layout = Layout::c_contiguous([1, 1, 3, 3]).expect("input layout");
        let output_layout = Layout::c_contiguous([1, 1, 2, 2]).expect("output layout");
        let input = Buffer(9);
        let output = Buffer(4);
        let error = plan_pooling_forward::<f32, _, 4, 2>(
            &PoolingForwardOperands {
                input: StridedView::new(&input, &input_layout),
                output: StridedView::new(&output, &output_layout),
            },
            WindowParameters::new([2, 2], [1, 1], [0, 0], [1, 1])
                .expect("valid pooling parameters"),
            true,
        )
        .expect_err("aliased pooling operands must be rejected");

        assert!(matches!(
            error,
            crate::HephaestusError::InvalidConfiguration { .. }
        ));
    }
}
