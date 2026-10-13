//! Strided launch metadata builders.

use super::types::StridedOperand;
use super::validate::validate_out;
use super::{StridedMeta, map_layout_err, pad_shape, pad_strides, to_u32};
use hephaestus_core::{DeviceBuffer, Result};

/// Validate and broadcast a binary strided operand triple, producing the
/// launch metadata; `None` when the dispatch is empty.
pub(crate) fn binary_strided_meta<T, const N: usize>(
    a: &StridedOperand<'_, T, N>,
    b: &StridedOperand<'_, T, N>,
    out: &StridedOperand<'_, T, N>,
) -> Result<Option<(StridedMeta, usize)>> {
    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    let b_layout = b
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len())
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.buffer.len())
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(None);
    }
    Ok(Some((
        StridedMeta {
            shape: pad_shape(out_layout.shape())?,
            a_strides: pad_strides(a_layout.strides())?,
            b_strides: pad_strides(b_layout.strides())?,
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(a_layout.offset(), "input offset")?,
                to_u32(b_layout.offset(), "input offset")?,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
            ],
        },
        len,
    )))
}

/// Validate and broadcast a unary strided operand pair, producing the launch
/// metadata; `None` when the dispatch is empty.
pub(crate) fn unary_strided_meta<T, const N: usize>(
    a: &StridedOperand<'_, T, N>,
    out: &StridedOperand<'_, T, N>,
) -> Result<Option<(StridedMeta, usize)>> {
    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len())
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(None);
    }
    Ok(Some((
        StridedMeta {
            shape: pad_shape(out_layout.shape())?,
            a_strides: pad_strides(a_layout.strides())?,
            b_strides: [0; 8],
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(a_layout.offset(), "input offset")?,
                0,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
            ],
        },
        len,
    )))
}

/// Validate a broadcast-scalar operand pair and build its dispatch metadata,
/// or `None` when the logical output is empty.
///
/// # Errors
///
/// Returns a layout validation failure.
pub(crate) fn scalar_strided_meta<T, const N: usize>(
    a: &StridedOperand<'_, T, N>,
    out: &StridedOperand<'_, T, N>,
) -> Result<Option<(StridedMeta, usize)>>
where
    T: eunomia::Pod,
{
    let out_layout = out.layout;
    let a_layout = a
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(a.buffer.len())
        .map_err(map_layout_err)?;
    let len = validate_out(out.buffer, out_layout)?;
    if len == 0 {
        return Ok(None);
    }
    Ok(Some((
        StridedMeta {
            shape: pad_shape(out_layout.shape())?,
            a_strides: pad_strides(a_layout.strides())?,
            b_strides: [0; 8],
            out_strides: pad_strides(out_layout.strides())?,
            offsets: [
                to_u32(a_layout.offset(), "input offset")?,
                0,
                to_u32(out_layout.offset(), "output offset")?,
                to_u32(len, "dispatch size")?,
            ],
        },
        len,
    )))
}
