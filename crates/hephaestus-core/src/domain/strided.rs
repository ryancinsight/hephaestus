//! Device-agnostic strided kernel metadata and layout packing.
//!
//! Strided dispatch over N-dimensional layouts uses fixed-rank (8) packed
//! metadata to avoid runtime malloc and kernel-side branching on rank.
//! This module defines the common metadata structure and layout-packing
//! helpers shared by all backends.

use crate::{HephaestusError, Result};

/// Maximum rank the packed rank-eight metadata covers.
///
/// Layouts with rank less than [`MAX_STRIDED_RANK`] are padded with
/// leading size-1 / stride-0 dimensions during packing, contributing
/// nothing to index computation.
pub const MAX_STRIDED_RANK: usize = 8;

/// Metadata passed to strided kernels, packed for rank-8 dispatch.
///
/// Every backend encodes this on the device and uses its contents to decode
/// linear output indices into per-operand offsets, so transposed, sliced, and
/// broadcast inputs execute directly from their device storage without a host
/// materialization copy.
#[repr(C)]
#[derive(Clone, Copy, Debug, eunomia::Pod, eunomia::Zeroable)]
pub struct StridedMeta {
    /// Logical shape, padded to rank 8 with leading 1s.
    pub shape: [u32; 8],
    /// Input-A element strides, padded to rank 8 with leading 0s.
    pub a_strides: [i32; 8],
    /// Input-B element strides, padded to rank 8 with leading 0s.
    /// (Zero for unary and scalar operations.)
    pub b_strides: [i32; 8],
    /// Output element strides, padded to rank 8 with leading 0s.
    pub out_strides: [i32; 8],
    /// [a_offset, b_offset, out_offset, dispatch_len]
    pub offsets: [u32; 4],
}

/// Convert a [`leto::LetoError`] to a [`HephaestusError`] dispatch failure.
#[must_use]
#[inline]
pub fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    }
}

/// Convert a `usize` to `u32`, or return a dispatch error if it overflows.
#[inline]
pub fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

/// Pad a fixed-rank shape array to rank-8 with leading 1s.
///
/// # Errors
///
/// Returns a dispatch error if any dimension exceeds `u32::MAX`.
#[inline]
pub fn pad_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; 8]> {
    let mut out = [1u32; 8];
    for (d, &dim) in shape.iter().enumerate() {
        out[8 - N + d] = to_u32(dim, "dimension")?;
    }
    Ok(out)
}

/// Pad a runtime-rank shape slice to rank-8 with leading 1s.
///
/// # Errors
///
/// Returns a dispatch error if any dimension exceeds `u32::MAX`.
#[inline]
pub fn pad_shape_dyn(shape: &[usize]) -> Result<[u32; 8]> {
    let mut out = [1u32; 8];
    for (d, &dim) in shape.iter().enumerate() {
        out[8 - shape.len() + d] = to_u32(dim, "dimension")?;
    }
    Ok(out)
}

/// Pad a fixed-rank stride array to rank-8 with leading 0s.
///
/// # Errors
///
/// Returns a dispatch error if any stride exceeds `i32::MAX` or is less than `i32::MIN`.
#[inline]
pub fn pad_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; 8]> {
    let mut out = [0i32; 8];
    for (d, &stride) in strides.iter().enumerate() {
        out[8 - N + d] = i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("stride {stride} exceeds i32 range"),
        })?;
    }
    Ok(out)
}

/// Pad a runtime-rank usize stride slice to rank-8 with leading 0s.
///
/// # Errors
///
/// Returns a dispatch error if any stride exceeds `i32::MAX`.
#[inline]
pub fn pad_usize_strides_dyn(strides: &[usize]) -> Result<[i32; 8]> {
    let mut out = [0i32; 8];
    for (d, &stride) in strides.iter().enumerate() {
        out[8 - strides.len() + d] =
            i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("stride {stride} exceeds i32 range"),
            })?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_shape_left_aligns() {
        let shape = [3, 4];
        let padded = pad_shape(shape).expect("valid shape");
        assert_eq!(padded[6], 3);
        assert_eq!(padded[7], 4);
        for value in &padded[..6] {
            assert_eq!(*value, 1);
        }
    }

    #[test]
    fn pad_strides_left_aligns() {
        let strides = [4isize, 1];
        let padded = pad_strides(strides).expect("valid strides");
        assert_eq!(padded[6], 4);
        assert_eq!(padded[7], 1);
        for value in &padded[..6] {
            assert_eq!(*value, 0);
        }
    }

    #[test]
    fn to_u32_overflow() {
        let big = (u32::MAX as usize) + 1;
        assert!(to_u32(big, "test").is_err());
    }
}
