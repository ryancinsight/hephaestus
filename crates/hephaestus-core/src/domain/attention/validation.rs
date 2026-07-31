use leto::Layout;

use crate::domain::buffer::DeviceBuffer;
use crate::domain::error::{HephaestusError, Result};
use crate::domain::planning::map_layout_err;

pub(super) fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

pub(super) fn validate_readonly<T, B, const R: usize>(buffer: &B, layout: &Layout<R>) -> Result<()>
where
    B: DeviceBuffer<T>,
{
    layout
        .validate_storage_len(buffer.len())
        .map_err(map_layout_err)
}

pub(super) fn validate_writable<T, B>(buffer: &B, layout: &Layout<3>, name: &str) -> Result<()>
where
    B: DeviceBuffer<T>,
{
    validate_readonly::<T, _, 3>(buffer, layout)?;
    if !layout.is_injective().map_err(map_layout_err)? {
        return Err(invalid(format!(
            "attention {name} layout must map logical indices injectively"
        )));
    }
    Ok(())
}

pub(super) fn expect_shape(name: &str, actual: [usize; 3], expected: [usize; 3]) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "attention {name} shape {actual:?} must equal {expected:?}"
        )));
    }
    Ok(())
}

pub(super) fn max_offset<const R: usize>(layout: &Layout<R>) -> Result<usize> {
    layout
        .checked_min_max_offsets()
        .map(|(_, maximum)| maximum)
        .map_err(map_layout_err)
}
