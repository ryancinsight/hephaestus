//! Shared helpers for backend-neutral dispatch planning.
//!
//! Narrowing conversions and layout-error mapping used by the per-family
//! planners ([`super::scan`], [`super::reduction`]) so the conversions live
//! in one place rather than being re-declared per family and per backend.

use crate::domain::error::{HephaestusError, Result};

// The canonical layout-error mapping lives in [`crate::domain::strided`], beside
// the strided metadata it serves. Re-exported here so the per-family planners
// keep importing it from their shared planning hub rather than each carrying a
// byte-identical copy whose diagnostic text can drift.
pub(crate) use crate::domain::strided::map_layout_err;

/// Narrow a signed stride to `i32` with a typed error.
pub(crate) fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

/// Narrow an unsigned extent to `u32` with a typed error.
pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}
