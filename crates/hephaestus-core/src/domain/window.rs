//! Shared validation and geometry for spatial window operations.

use bytemuck::Pod;
use leto::{Layout, WindowParameters};

use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::planning::map_layout_err;

/// Validated batch, channel, and spatial geometry for a window operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowPlan<const S: usize> {
    /// Batch extent.
    pub batch: usize,
    /// Channel extent.
    pub channels: usize,
    /// Input spatial extents.
    pub input_spatial: [usize; S],
    /// Output spatial extents derived from [`WindowParameters`].
    pub output_spatial: [usize; S],
    /// Validated window parameters.
    pub parameters: WindowParameters<S>,
    /// Number of points in one spatial window.
    pub kernel_volume: usize,
    /// Number of output spatial locations.
    pub output_locations: usize,
    /// Logical input element count.
    pub input_elements: usize,
    /// Largest physical element offset touched by the validated operands.
    pub max_physical_offset: usize,
}

impl<const S: usize> WindowPlan<S> {
    /// Validate values narrowed by a backend address calculation.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when a geometry or
    /// physical offset exceeds `max_inclusive`.
    pub fn validate_address_limit(&self, max_inclusive: usize) -> Result<()> {
        let values = [
            self.batch,
            self.channels,
            self.kernel_volume,
            self.output_locations,
            self.input_elements,
            self.max_physical_offset,
        ];
        if values.into_iter().any(|value| value > max_inclusive)
            || self
                .input_spatial
                .into_iter()
                .chain(self.output_spatial)
                .chain(*self.parameters.kernel())
                .chain(*self.parameters.stride())
                .chain(*self.parameters.padding())
                .chain(*self.parameters.dilation())
                .any(|value| value > max_inclusive)
        {
            return Err(invalid(format!(
                "window plan exceeds backend address limit {max_inclusive}"
            )));
        }
        Ok(())
    }
}

/// Build validated window geometry from an input layout.
pub(crate) fn plan_window<T, B, const R: usize, const S: usize>(
    layout: &Layout<R>,
    buffer: &B,
    parameters: WindowParameters<S>,
) -> Result<WindowPlan<S>>
where
    T: Pod,
    B: DeviceBuffer<T>,
{
    validate_rank::<R, S>("window")?;
    layout
        .validate_storage_len(buffer.len())
        .map_err(map_layout_err)?;

    let shape = layout.shape();
    let Some(&batch) = shape.first() else {
        return Err(invalid("window input is missing its batch axis"));
    };
    let Some(&channels) = shape.get(1) else {
        return Err(invalid("window input is missing its channel axis"));
    };
    let Some(spatial_shape) = shape.get(2..) else {
        return Err(invalid("window input is missing its spatial axes"));
    };
    let mut input_spatial = [0; S];
    input_spatial.copy_from_slice(spatial_shape);
    let output_spatial = parameters
        .output_shape(input_spatial)
        .map_err(|error| invalid(error.to_string()))?;
    let output_locations = output_spatial
        .into_iter()
        .try_fold(1_usize, |product, extent| {
            product
                .checked_mul(extent)
                .ok_or_else(|| invalid("window output location count overflows"))
        })?;
    let input_elements = layout.checked_size().map_err(map_layout_err)?;

    Ok(WindowPlan {
        batch,
        channels,
        input_spatial,
        output_spatial,
        kernel_volume: parameters
            .kernel_volume()
            .map_err(|error| invalid(error.to_string()))?,
        parameters,
        output_locations,
        input_elements,
        max_physical_offset: max_offset(layout)?,
    })
}

pub(crate) fn validate_rank<const R: usize, const S: usize>(name: &str) -> Result<()> {
    let expected = S
        .checked_add(2)
        .ok_or_else(|| invalid(format!("{name} tensor rank overflows")))?;
    if S == 0 || R != expected {
        return Err(invalid(format!(
            "{name} tensor rank {R} must equal spatial rank {S} plus batch/channel axes"
        )));
    }
    Ok(())
}

pub(crate) fn validate_readonly<T, B, const R: usize>(layout: &Layout<R>, buffer: &B) -> Result<()>
where
    T: Pod,
    B: DeviceBuffer<T>,
{
    layout
        .validate_storage_len(buffer.len())
        .map_err(map_layout_err)
}

pub(crate) fn validate_writable<T, B, const R: usize>(
    layout: &Layout<R>,
    buffer: &B,
    name: &str,
) -> Result<()>
where
    T: Pod,
    B: DeviceBuffer<T>,
{
    validate_readonly::<T, B, R>(layout, buffer)?;
    if !layout.is_injective().map_err(map_layout_err)? {
        return Err(invalid(format!("window {name} layout must be injective")));
    }
    Ok(())
}

pub(crate) fn validate_shape<const R: usize>(
    actual: [usize; R],
    expected_batch: usize,
    expected_channels: usize,
    expected_spatial: &[usize],
    name: &str,
) -> Result<()> {
    let Some((&batch, rest)) = actual.split_first() else {
        return Err(invalid(format!("window {name} is missing its batch axis")));
    };
    let Some((&channels, spatial)) = rest.split_first() else {
        return Err(invalid(format!(
            "window {name} is missing its channel axis"
        )));
    };
    if batch != expected_batch || channels != expected_channels || spatial != expected_spatial {
        return Err(invalid(format!(
            "window {name} shape {actual:?} does not match [{expected_batch}, {expected_channels}, {expected_spatial:?}]"
        )));
    }
    Ok(())
}

pub(crate) fn max_offset<const R: usize>(layout: &Layout<R>) -> Result<usize> {
    layout
        .checked_min_max_offsets()
        .map(|(_, maximum)| maximum)
        .map_err(map_layout_err)
}

pub(crate) fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}
