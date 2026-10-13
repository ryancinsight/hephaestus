use hephaestus_core::{HephaestusError, Result, map_layout_err};
use leto::Layout;

use crate::infrastructure::buffer::WgpuBuffer;

#[inline]
pub(crate) fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

/// Validate an output layout against its buffer and return the logical length.
pub(crate) fn validate_out<T, const N: usize>(
    out: &WgpuBuffer<T>,
    out_layout: &Layout<N>,
) -> Result<usize> {
    out_layout
        .validate_storage_len(out.len)
        .map_err(map_layout_err)?;
    if !out_layout.is_injective().map_err(map_layout_err)? {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must be non-overlapping".to_string(),
        });
    }
    out_layout.checked_size().map_err(map_layout_err)
}
