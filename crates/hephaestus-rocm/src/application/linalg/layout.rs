//! Device-side rank-2 matrix layout metadata and width conversions for ROCm.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{HephaestusError, Result};
use leto::Layout;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuMatrixLayout {
    pub(super) shape: [u32; 2],
    pub(super) strides: [i32; 2],
    pub(super) offset: u32,
    pub(super) _pad: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<GpuMatrixLayout>() == 32);

#[inline]
pub(super) fn map_layout(layout: &Layout<2>) -> Result<GpuMatrixLayout> {
    Ok(GpuMatrixLayout {
        shape: [
            to_u32(layout.shape()[0], "dimension")?,
            to_u32(layout.shape()[1], "dimension")?,
        ],
        strides: [
            to_i32(layout.strides()[0], "stride")?,
            to_i32(layout.strides()[1], "stride")?,
        ],
        offset: to_u32(layout.offset(), "offset")?,
        _pad: [0; 3],
    })
}

#[inline]
pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

#[inline]
pub(crate) fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

#[inline]
pub(super) fn to_i64(value: isize, what: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i64 range"),
    })
}
