//! Packed HIP metadata and checked layout conversion.

use crate::application::strided::StridedOperand;
use eunomia::{Pod, Zeroable};
use hephaestus_core::{DeviceBuffer, HephaestusError, Result};

/// Maximum rank represented by the packed strided metadata.
pub const MAX_STRIDED_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct StridedMeta {
    pub(crate) shape: [u32; MAX_STRIDED_RANK],
    pub(crate) a_strides: [i32; MAX_STRIDED_RANK],
    pub(crate) b_strides: [i32; MAX_STRIDED_RANK],
    pub(crate) out_strides: [i32; MAX_STRIDED_RANK],
    pub(crate) offsets: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<StridedMeta>() == (4 * MAX_STRIDED_RANK + 4) * 4);
const _: () = assert!(core::mem::align_of::<StridedMeta>() == 4);

impl StridedMeta {
    pub(crate) fn new<const N: usize>(
        first: &leto::Layout<N>,
        second: Option<&leto::Layout<N>>,
        output: &leto::Layout<N>,
        len: usize,
    ) -> Result<Self> {
        let offset = |value| {
            u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("layout offset {value} exceeds u32 range"),
            })
        };
        Ok(Self {
            shape: pad_shape(output.shape())?,
            a_strides: pad_strides(first.strides())?,
            b_strides: second.map_or(Ok([0; MAX_STRIDED_RANK]), |layout| {
                pad_strides(layout.strides())
            })?,
            out_strides: pad_strides(output.strides())?,
            offsets: [
                offset(first.offset())?,
                second.map_or(Ok(0), |layout| offset(layout.offset()))?,
                offset(output.offset())?,
                dispatch_len(len)?,
            ],
        })
    }
}

#[cfg(test)]
mod tests;

pub(crate) fn check_rank<const N: usize>() -> Result<()> {
    if N > MAX_STRIDED_RANK {
        return Err(HephaestusError::DispatchFailed {
            message: format!("strided dispatch rank {N} exceeds maximum {MAX_STRIDED_RANK}"),
        });
    }
    Ok(())
}

pub(crate) fn pad_shape<const N: usize>(shape: [usize; N]) -> Result<[u32; MAX_STRIDED_RANK]> {
    check_rank::<N>()?;
    let mut padded = [1_u32; MAX_STRIDED_RANK];
    for (axis, &extent) in shape.iter().enumerate() {
        padded[MAX_STRIDED_RANK - N + axis] =
            u32::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("dimension {extent} exceeds u32 range"),
            })?;
    }
    Ok(padded)
}

pub(crate) fn pad_strides<const N: usize>(strides: [isize; N]) -> Result<[i32; MAX_STRIDED_RANK]> {
    check_rank::<N>()?;
    let mut padded = [0_i32; MAX_STRIDED_RANK];
    for (axis, &stride) in strides.iter().enumerate() {
        padded[MAX_STRIDED_RANK - N + axis] =
            i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("stride {stride} exceeds i32 range"),
            })?;
    }
    Ok(padded)
}

fn validate_output<T: Pod, const N: usize>(output: StridedOperand<'_, T, N>) -> Result<usize> {
    if output.layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    output
        .layout
        .validate_storage_len(output.buffer.len())
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("layout rejected: {error}"),
        })?;
    output
        .layout
        .checked_size()
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("layout rejected: {error}"),
        })
}

pub(crate) fn dispatch_len(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("strided dispatch length {len} exceeds u32 range"),
    })
}

pub(crate) fn map_layout_err(error: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {error}"),
    }
}

/// Validate, broadcast, and alias-check a binary strided operand triple,
/// producing the launch metadata; `None` when the dispatch is empty.
pub(crate) fn binary_strided_meta<T, const N: usize>(
    lhs: &StridedOperand<'_, T, N>,
    rhs: &StridedOperand<'_, T, N>,
    output: &StridedOperand<'_, T, N>,
) -> Result<Option<(StridedMeta, usize)>>
where
    T: Pod,
{
    check_rank::<N>()?;
    let lhs_layout = lhs
        .layout
        .broadcast(output.layout.shape())
        .map_err(map_layout_err)?;
    let rhs_layout = rhs
        .layout
        .broadcast(output.layout.shape())
        .map_err(map_layout_err)?;
    lhs_layout
        .validate_storage_len(lhs.buffer.len())
        .map_err(map_layout_err)?;
    rhs_layout
        .validate_storage_len(rhs.buffer.len())
        .map_err(map_layout_err)?;
    if lhs.buffer.aliases(output.buffer) || rhs.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias either input buffer".to_string(),
        });
    }
    let len = validate_output(*output)?;
    if len == 0 {
        return Ok(None);
    }
    let meta = StridedMeta::new(&lhs_layout, Some(&rhs_layout), output.layout, len)?;
    Ok(Some((meta, len)))
}

/// Validate, broadcast, and alias-check a unary strided operand pair,
/// producing the launch metadata; `None` when the dispatch is empty.
pub(crate) fn unary_strided_meta<T, const N: usize>(
    input: &StridedOperand<'_, T, N>,
    output: &StridedOperand<'_, T, N>,
) -> Result<Option<(StridedMeta, usize)>>
where
    T: Pod,
{
    check_rank::<N>()?;
    let input_layout = input
        .layout
        .broadcast(output.layout.shape())
        .map_err(map_layout_err)?;
    input_layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_output(*output)?;
    if len == 0 {
        return Ok(None);
    }
    let meta = StridedMeta::new(&input_layout, None, output.layout, len)?;
    Ok(Some((meta, len)))
}

/// Validate a broadcast-scalar operand pair and build its dispatch metadata,
/// or `None` when the logical output is empty.
///
/// # Errors
///
/// Returns a layout validation failure or an aliasing violation.
pub(crate) fn scalar_strided_meta<T, const N: usize>(
    input: &StridedOperand<'_, T, N>,
    output: &StridedOperand<'_, T, N>,
) -> Result<Option<(StridedMeta, usize)>>
where
    T: Pod,
{
    check_rank::<N>()?;
    let input_layout = input
        .layout
        .broadcast(output.layout.shape())
        .map_err(map_layout_err)?;
    input_layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias input buffer".to_string(),
        });
    }
    let len = validate_output(*output)?;
    if len == 0 {
        return Ok(None);
    }
    let meta = StridedMeta::new(&input_layout, None, output.layout, len)?;
    Ok(Some((meta, len)))
}
